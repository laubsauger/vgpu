// Procedural meshes, built once on the CPU. Every part mesh is one merged vertex buffer whose
// vertices carry a material code, so a robot part (a light shell over a slate frame with a metal
// actuator) is a single instanced draw. Leg segments run along +Y from the joint at 0 to the next
// joint at the bone length, matching the rig locals; the hinge axis is local +X. Body space is +Z
// forward, +Y up, +X to the robot's right. Right-side hips and femurs face their accent caps toward +X;
// the left meshes are the same geometry mirrored in X.
//
// Vertex layout (shared with the terrain): position (3), normal (3), material (1) floats.

import { FEMUR, HIP_OFFSET, TIBIA } from "./leg";
import { FOOT_RADIUS } from "./robot";
import { HALF } from "./terrain";

export const MESH_VERTEX_FLOATS = 7;

/** Material codes; keep in sync with the MATERIAL_* constants in common.wgsl. */
export const MATERIAL = {
  cover: 0,
  accent: 1,
  graphite: 2,
  lens: 3,
  led: 4,
  rubber: 5,
  metal: 6,
  plinth: 7,
} as const;

export interface MeshData {
  readonly vertices: Float32Array<ArrayBuffer>;
  readonly indices: Uint32Array<ArrayBuffer>;
}

type V3 = readonly [number, number, number];

/** A local frame: the primitive's x/y/z axes and origin in mesh space. */
interface Frame {
  readonly x: V3;
  readonly y: V3;
  readonly z: V3;
  readonly origin: V3;
}

const AXES: Frame = { x: [1, 0, 0], y: [0, 1, 0], z: [0, 0, 1], origin: [0, 0, 0] };

function at(origin: V3): Frame {
  return { ...AXES, origin };
}

/** A frame whose +Y runs along local +X (actuators and hinge pins across a leg). */
function across(origin: V3): Frame {
  return { x: [0, -1, 0], y: [1, 0, 0], z: [0, 0, 1], origin };
}

/** A frame whose +Y runs along +Z (forward). */
function forward(origin: V3): Frame {
  return { x: [1, 0, 0], y: [0, 0, 1], z: [0, -1, 0], origin };
}

/** One cross-section of a sweep: centre (x, y, z) and elliptical half widths across X and Z. */
interface Station {
  readonly x: number;
  readonly y: number;
  readonly z: number;
  readonly rx: number;
  readonly rz: number;
}

class Builder {
  private readonly vertices: number[] = [];
  private readonly indices: number[] = [];

  private vertex(frame: Frame, p: V3, n: V3, material: number): number {
    const { x, y, z, origin } = frame;
    const nx = x[0] * n[0] + y[0] * n[1] + z[0] * n[2];
    const ny = x[1] * n[0] + y[1] * n[1] + z[1] * n[2];
    const nz = x[2] * n[0] + y[2] * n[1] + z[2] * n[2];
    const length = Math.hypot(nx, ny, nz) || 1;
    this.vertices.push(
      origin[0] + x[0] * p[0] + y[0] * p[1] + z[0] * p[2],
      origin[1] + x[1] * p[0] + y[1] * p[1] + z[1] * p[2],
      origin[2] + x[2] * p[0] + y[2] * p[1] + z[2] * p[2],
      nx / length,
      ny / length,
      nz / length,
      material,
    );
    return this.vertices.length / MESH_VERTEX_FLOATS - 1;
  }

  /** Counter-clockwise quads over a (rows+1)×(columns+1) vertex grid starting at `first`. */
  private grid(first: number, rows: number, columns: number): void {
    const stride = columns + 1;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < columns; c++) {
        const a = first + r * stride + c;
        const b = a + 1;
        const d = a + stride;
        const e = d + 1;
        this.indices.push(a, b, d, b, e, d);
      }
    }
  }

  /** Ellipsoid with `radii` and an optional radial `shape(u, v)` multiplier (u around, v up). */
  ellipsoid(frame: Frame, radii: V3, material: number, segments = 18, rings = 12, shape?: (u: number, v: number) => number): this {
    const first = this.vertices.length / MESH_VERTEX_FLOATS;
    for (let r = 0; r <= rings; r++) {
      const v = r / rings;
      const phi = Math.PI * (v - 0.5);
      for (let s = 0; s <= segments; s++) {
        const u = s / segments;
        const theta = u * Math.PI * 2;
        const k = shape ? shape(u % 1, v) : 1;
        const ux = Math.cos(phi) * Math.sin(theta);
        const uy = Math.sin(phi);
        const uz = Math.cos(phi) * Math.cos(theta);
        // The normal of an ellipsoid is the unit direction scaled by 1/radii.
        this.vertex(frame, [ux * radii[0] * k, uy * radii[1] * k, uz * radii[2] * k], [ux / radii[0], uy / radii[1], uz / radii[2]], material);
      }
    }
    this.grid(first, rings, segments);
    return this;
  }

  /**
   * Superellipsoid: `squareness` 1 is an ellipsoid, smaller values flatten the faces toward a
   * rounded box (machined panels rather than organic shells).
   */
  rounded(frame: Frame, radii: V3, squareness: number, material: number, segments = 28, rings = 16): this {
    const first = this.vertices.length / MESH_VERTEX_FLOATS;
    const e = squareness;
    // Snap trig residuals (sin 2π ≈ −2.4e-16) to zero first: a fractional power lifts them to a few
    // thousandths, which opens the seam column and the poles into visible slits.
    const spow = (value: number, exponent: number) => (Math.abs(value) < 1e-12 ? 0 : Math.sign(value) * Math.abs(value) ** exponent);
    for (let r = 0; r <= rings; r++) {
      const phi = Math.PI * (r / rings - 0.5);
      const cp = Math.cos(phi);
      const sp = Math.sin(phi);
      for (let s = 0; s <= segments; s++) {
        const theta = (s / segments) * Math.PI * 2;
        const st = Math.sin(theta);
        const ct = Math.cos(theta);
        this.vertex(
          frame,
          [radii[0] * spow(cp, e) * spow(st, e), radii[1] * spow(sp, e), radii[2] * spow(cp, e) * spow(ct, e)],
          [(spow(cp, 2 - e) * spow(st, 2 - e)) / radii[0], spow(sp, 2 - e) / radii[1], (spow(cp, 2 - e) * spow(ct, 2 - e)) / radii[2]],
          material,
        );
      }
    }
    this.grid(first, rings, segments);
    return this;
  }

  /** Tapered tube along +Y from y0 (radius r0) to y1 (radius r1), with flat caps. */
  tube(frame: Frame, y0: number, y1: number, r0: number, r1: number, material: number, sides = 12): this {
    return this.sweep(
      frame,
      [
        { x: 0, y: y0, z: 0, rx: r0, rz: r0 },
        { x: 0, y: y1, z: 0, rx: r1, rz: r1 },
      ],
      material,
      sides,
    );
  }

  /**
   * Capped sweep through `stations` stacked along +Y, each an ellipse in the plane across the spine
   * (local X and the spine-perpendicular in YZ). Bowed, tapered limbs and covers.
   */
  sweep(frame: Frame, stations: readonly Station[], material: number, sides = 16): this {
    const first = this.vertices.length / MESH_VERTEX_FLOATS;
    const across: [number, number][] = [];
    for (let k = 0; k < stations.length; k++) {
      const previous = stations[Math.max(0, k - 1)]!;
      const next = stations[Math.min(stations.length - 1, k + 1)]!;
      const ty = next.y - previous.y;
      const tz = next.z - previous.z;
      const length = Math.hypot(ty, tz) || 1;
      // X × tangent: the in-plane axis perpendicular to the spine (+Z for a spine along +Y).
      across.push([-tz / length, ty / length]);
    }
    const ring = (k: number, s: number): V3 => {
      const station = stations[k]!;
      const [by, bz] = across[k]!;
      const theta = (s / sides) * Math.PI * 2;
      const sx = Math.sin(theta) * station.rx;
      const sz = Math.cos(theta) * station.rz;
      return [station.x + sx, station.y + by * sz, station.z + bz * sz];
    };
    for (let k = 0; k < stations.length; k++) {
      const station = stations[k]!;
      const [by, bz] = across[k]!;
      for (let s = 0; s <= sides; s++) {
        const theta = (s / sides) * Math.PI * 2;
        const nx = Math.sin(theta) / station.rx;
        const nb = Math.cos(theta) / station.rz;
        this.vertex(frame, ring(k, s), [nx, by * nb, bz * nb], material);
      }
    }
    this.grid(first, stations.length - 1, sides);
    for (const [k, outward] of [
      [0, -1],
      [stations.length - 1, 1],
    ] as const) {
      const station = stations[k]!;
      const [by, bz] = across[k]!;
      // Cap normal: ± the spine tangent (perpendicular to `across` in YZ).
      const normal: V3 = [0, bz * outward, -by * outward];
      const centre = this.vertex(frame, [station.x, station.y, station.z], normal, material);
      for (let s = 0; s <= sides; s++) this.vertex(frame, ring(k, s), normal, material);
      for (let s = 0; s < sides; s++) {
        if (outward > 0) this.indices.push(centre, centre + 1 + s, centre + 2 + s);
        else this.indices.push(centre, centre + 2 + s, centre + 1 + s);
      }
    }
    return this;
  }

  /** Open box sides from y0 to y1 with half extent `half`, outward normals (the plinth). */
  walls(half: number, y0: number, y1: number, material: number): this {
    const sides: [V3, V3][] = [
      [[1, 0, 0], [0, 0, -1]],
      [[-1, 0, 0], [0, 0, 1]],
      [[0, 0, 1], [1, 0, 0]],
      [[0, 0, -1], [-1, 0, 0]],
    ];
    for (const [n, t] of sides) {
      const first = this.vertices.length / MESH_VERTEX_FLOATS;
      for (const y of [y0, y1]) {
        for (const side of [-1, 1]) {
          this.vertex(AXES, [n[0] * half + t[0] * half * side, y, n[2] * half + t[2] * half * side], n, material);
        }
      }
      this.indices.push(first, first + 1, first + 2, first + 1, first + 3, first + 2);
    }
    return this;
  }

  build(): MeshData {
    return { vertices: new Float32Array(this.vertices), indices: new Uint32Array(this.indices) };
  }
}

/** The same mesh reflected in X, with the winding flipped so it stays front-facing. */
export function mirrorX(mesh: MeshData): MeshData {
  const vertices = mesh.vertices.slice();
  for (let v = 0; v < vertices.length; v += MESH_VERTEX_FLOATS) {
    vertices[v] = -vertices[v]!;
    vertices[v + 3] = -vertices[v + 3]!;
  }
  const indices = mesh.indices.slice();
  for (let i = 0; i < indices.length; i += 3) {
    const b = indices[i + 1]!;
    indices[i + 1] = indices[i + 2]!;
    indices[i + 2] = b;
  }
  return { vertices, indices };
}

/** Linear interpolation of station rows: `count` stations from t = 0 to 1. */
function stations(count: number, at: (t: number) => Station): Station[] {
  return Array.from({ length: count }, (_, k) => at(k / (count - 1)));
}

const mix = (a: number, b: number, t: number) => a + (b - a) * t;

export type PartMeshes = Record<"shell" | "payload" | "hipRight" | "hipLeft" | "femurRight" | "femurLeft" | "tibia" | "foot", MeshData>;

/** Torso half extents, body space (the torso is centred a little behind the body origin). */
export const CHASSIS: V3 = [0.118, 0.07, 0.32];
const TORSO_Z = -0.04;

export function buildPartMeshes(): PartMeshes {
  const { cover, accent, graphite, lens, led, rubber, metal } = MATERIAL;
  const [cx, cy, cz] = CHASSIS;
  // Torso: a soft, pill-shaped light shell over a slate keel, with an accent spine ridge on top, a
  // vented flank panel each side, and a rounded head pod on a short neck at the front.
  const shell = new Builder()
    .rounded(at([0, 0, TORSO_Z]), [cx, cy, cz], 0.55, cover, 36, 20)
    .rounded(at([0, -0.04, TORSO_Z]), [cx - 0.012, 0.045, cz - 0.03], 0.45, graphite, 28, 14)
    .rounded(at([0, cy - 0.004, TORSO_Z - 0.02]), [0.026, 0.014, cz - 0.08], 0.35, accent, 20, 10);
  for (const side of [-1, 1]) {
    // Flank panel with three vent slots, and a round service hatch behind it.
    shell.rounded(at([side * (cx - 0.004), 0.004, TORSO_Z + 0.02]), [0.01, 0.03, 0.12], 0.3, graphite, 14, 8);
    for (const z of [-0.04, 0.02, 0.08]) shell.rounded(at([side * (cx + 0.005), 0.004, TORSO_Z + z]), [0.003, 0.018, 0.008], 0.3, accent, 8, 6);
    shell.tube(across([side > 0 ? cx - 0.02 : -cx - 0.002, 0.006, TORSO_Z - 0.2]), 0, 0.022, 0.02, 0.02, graphite, 16);
  }
  // Neck collar and head pod: a wraparound dark visor with two LED eyes (brighter on payload robots)
  // and a slate chin.
  shell
    .tube(forward([0, 0.026, TORSO_Z + cz - 0.06]), 0, 0.12, 0.05, 0.044, graphite, 22)
    .rounded(at([0, 0.046, 0.372]), [0.084, 0.058, 0.076], 0.7, cover, 28, 16)
    .rounded(at([0, 0.054, 0.4]), [0.077, 0.023, 0.052], 0.55, lens, 24, 10)
    .rounded(at([0, 0.0, 0.392]), [0.058, 0.016, 0.05], 0.6, graphite, 18, 8);
  for (const side of [-1, 1]) shell.ellipsoid(at([side * 0.032, 0.055, 0.449]), [0.012, 0.0075, 0.006], led, 12, 6);
  // Tail: a slate cap with an accent ring and a rear lens.
  shell
    .tube(forward([0, 0.002, TORSO_Z - cz - 0.004]), 0, 0.03, 0.04, 0.05, graphite, 20)
    .tube(forward([0, 0.002, TORSO_Z - cz + 0.018]), 0, 0.012, 0.053, 0.053, accent, 20)
    .ellipsoid(at([0, 0.002, TORSO_Z - cz - 0.006]), [0.014, 0.014, 0.006], lens, 12, 6);

  // Payload: a slim cargo deck with side rails, a dome antenna housing at the back, a gimbal camera
  // at the front and a whip antenna with a lit tip. Origin on the spine ridge (rig PAYLOAD_OFFSET).
  const payload = new Builder().rounded(at([0, 0.016, 0]), [0.094, 0.016, 0.24], 0.25, graphite, 28, 10);
  for (const side of [-1, 1]) payload.tube(forward([side * 0.088, 0.036, -0.22]), 0, 0.44, 0.007, 0.007, metal, 10);
  payload
    .tube(at([0, 0, -0.12]), 0.03, 0.05, 0.068, 0.068, accent, 28)
    .ellipsoid(at([0, 0.05, -0.12]), [0.064, 0.05, 0.064], cover, 28, 14)
    .tube(at([0, 0, 0.17]), 0.03, 0.07, 0.012, 0.012, metal, 10)
    .ellipsoid(at([0, 0.088, 0.17]), [0.03, 0.03, 0.03], graphite, 20, 12)
    .ellipsoid(at([0, 0.088, 0.196]), [0.014, 0.014, 0.008], lens, 12, 6)
    .tube(at([-0.06, 0, -0.2]), 0.03, 0.22, 0.004, 0.003, graphite, 8)
    .ellipsoid(at([-0.06, 0.224, -0.2]), [0.007, 0.007, 0.007], led, 10, 6);

  // Hip (right): a slate roll housing at the pivot and the pitch motor out along X to the femur
  // hinge, banded in the accent colour.
  const hip = new Builder()
    .rounded(at([0.01, 0, 0]), [0.046, 0.046, 0.06], 0.6, graphite, 22, 12)
    .tube(across([0, 0, 0]), 0.02, HIP_OFFSET - 0.03, 0.044, 0.044, metal, 22)
    .tube(across([0, 0, 0]), HIP_OFFSET - 0.036, HIP_OFFSET - 0.026, 0.049, 0.049, accent, 22);

  // Femur (right): a slate hub over the pitch motor with an accent cap, a straight round strut in
  // the light shell, a metal push rod down its back to the knee, and a knee hub with an accent cap.
  const femur = new Builder()
    .tube(across([-0.03, 0, 0]), 0, 0.06, 0.042, 0.042, graphite, 22)
    .tube(across([0.03, 0, 0]), 0, 0.008, 0.038, 0.032, accent, 22)
    .sweep(
      AXES,
      stations(6, (t) => ({ x: 0.004, y: mix(0.03, FEMUR - 0.025, t), z: 0.004, rx: mix(0.025, 0.019, t), rz: mix(0.034, 0.024, t) })),
      cover,
      18,
    )
    .tube(at([-0.004, 0, -0.036]), 0.02, FEMUR - 0.03, 0.0065, 0.0065, metal, 10)
    .tube(across([-0.026, FEMUR, 0]), 0, 0.052, 0.026, 0.026, graphite, 18)
    .tube(across([0.026, FEMUR, 0]), 0, 0.006, 0.019, 0.015, accent, 16);

  // Tibia: a slate knee block and a straight, tapering light-shell shin with an accent ankle collar
  // over a slate ankle.
  const tibia = new Builder()
    .rounded(at([0, 0.02, -0.004]), [0.022, 0.04, 0.03], 0.55, graphite, 18, 10)
    .sweep(
      AXES,
      stations(5, (t) => ({ x: 0, y: mix(0.04, TIBIA - FOOT_RADIUS * 1.6, t), z: 0, rx: mix(0.018, 0.0125, t), rz: mix(0.022, 0.0135, t) })),
      cover,
      16,
    )
    .tube(AXES, TIBIA - FOOT_RADIUS * 1.75, TIBIA - FOOT_RADIUS * 1.35, 0.0155, 0.0155, accent, 14)
    .tube(AXES, TIBIA - FOOT_RADIUS * 1.4, TIBIA - FOOT_RADIUS * 0.4, 0.012, 0.016, graphite, 14);

  // Foot: a rubber ball whose bottom touches the ground one FOOT_RADIUS below the foot point.
  const foot = new Builder().ellipsoid(AXES, [FOOT_RADIUS, FOOT_RADIUS, FOOT_RADIUS], rubber, 20, 12);

  const hipMesh = hip.build();
  const femurMesh = femur.build();
  return {
    shell: shell.build(),
    payload: payload.build(),
    hipRight: hipMesh,
    hipLeft: mirrorX(hipMesh),
    femurRight: femurMesh,
    femurLeft: mirrorX(femurMesh),
    tibia: tibia.build(),
    foot: foot.build(),
  };
}

/** Plinth depth below the tile rim. */
export const PLINTH_DEPTH = 0.9;

/** The training ground's concrete plinth walls. */
export function buildPlinthMesh(): MeshData {
  return new Builder().walls(HALF, -PLINTH_DEPTH, 0.002, MATERIAL.plinth).build();
}
