/*
    voxel_flight_simulator - A simple game where you fly around randomly generated, recursive, voxel worlds.
    Copyright (C) 2023 Ryan Andersen

    voxel_flight_simulator is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    voxel_flight_simulator is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with voxel_flight_simulator. If not, see <https://www.gnu.org/licenses/>.
*/

#version 450
precision highp float;
layout (location = 0) in vec2 coord;

layout (location = 0) out vec4 fragColor;

layout (push_constant) uniform Push {
	vec3 camera_position;
	float time;
	vec4 camera_quaternion;
	vec3 light_dir;
	float aspect_ratio;
} push;

struct Voxel {
	vec4 averageColour;
	uint ftl;
	uint ftr;
	uint fbl;
	uint fbr;
	uint btl;
	uint btr;
	uint bbl;
	uint bbr;
	uint vtype;
};
const uint emptyVoxel = 0xFFFFFFFF;

layout(set = 0, binding = 0) readonly buffer VoxelOctree {   
	Voxel voxels[];
} voxelOctree;

const float pi = 3.14159265358;
const float e = 2.718281828;
const int maxIterations = 35;
const float maxIterationsF = float(maxIterations);
const int globalMaxDepth = 15;
const float epsilon = 0.005;
const float unitEpsilon = 1.001;
const vec3 dirX = vec3(1.0, 0.0, 0.0);
const vec3 dirY = vec3(0.0, 1.0, 0.0);
const vec3 dirZ = vec3(0.0, 0.0, 1.0);
const vec3 negDirX = vec3(-1.0, 0.0, 0.0);
const vec3 negDirY = vec3(0.0, -1.0, 0.0);
const vec3 negDirZ = vec3(0.0, 0.0, -1.0);

const float goalRadiusSquared = 0.75;

// Phong lighting
const vec4 fogColour = vec4(0.42, 0.525, 0.45, 1.0);
const vec4 skyColour = vec4(0.08, 0.2, 0.75, 1.0);
const vec4 groundColour = vec4(0.2, 0.08, 0.08, 1.0);
const float ambientStrength = 0.5;
const vec3 lightColor = vec3(0.85);
const vec3 ambientLight = ambientStrength * lightColor;

const vec3 ftlCell = vec3(-0.5, 0.5, -0.5);
const vec3 ftrCell = vec3(0.5, 0.5, -0.5);
const vec3 fblCell = vec3(-0.5, -0.5, -0.5);
const vec3 fbrCell = vec3(0.5, -0.5, -0.5);
const vec3 btlCell = vec3(-0.5, 0.5, 0.5);
const vec3 btrCell = vec3(0.5, 0.5, 0.5);
const vec3 bblCell = vec3(-0.5, -0.5, 0.5);
const vec3 bbrCell = vec3(0.5, -0.5, 0.5);

vec3 rotateByQuaternion(vec3 v, vec4 q) {
	vec3 temp = cross(q.xyz, cross(q.xyz, v) + q.w * v);
	return v + temp+temp;
}

vec3 cubeNorm(vec3 t) {
	vec3 s = abs(t);
	if(s.x >= s.y && s.x >= s.z) {
		return vec3(sign(t.x), 0.0, 0.0);
	} else if(s.y >= s.x && s.y >= s.z) {
		return vec3(0.0, sign(t.y), 0.0);
	} else {
		return vec3(0.0, 0.0, sign(t.z));
	}
}

vec3 projectToOutsideDistance(vec3 t) {
	vec3 s = abs(t);
	if(s.x >= s.y && s.x >= s.z) {
		return vec3(sign(t.x) * unitEpsilon - t.x, 0.0, 0.0);
	} else if(s.y >= s.x && s.y >= s.z) {
		return vec3(0.0, sign(t.y) * unitEpsilon - t.y, 0.0);
	} else {
		return vec3(0.0, 0.0, sign(t.z) * unitEpsilon - t.z);
	}
}

bool insideCube(vec3 t) {
	t = abs(t);
	return t.x <= 1.0 && t.y <= 1.0 && t.z <= 1.0;
}

bool projectToRootVoxel(inout vec3 p, vec3 d, vec3 invD) {
	if(insideCube(p)) return true;

	vec3 s;
	float t;
	if(abs(p.x) > 1.0) {
		t = (sign(p.x) - p.x)*invD.x;
		if(t >= 0.0) {
			s = p + t*d;
			if(abs(s.y) <= 1.0 && abs(s.z) <= 1.0) {
				p = s;
				return true;
			}
		}
	}
	if(abs(p.y) > 1.0) {
		t = (sign(p.y) - p.y)*invD.y;
		if(t >= 0.0) {
			s = p + t*d;
			if(abs(s.x) <= 1.0 && abs(s.z) <= 1.0) {
				p = s;
				return true;
			}
		}
	}
	if(abs(p.z) > 1.0) {
		t = (sign(p.z) - p.z)*invD.z;
		if(t >= 0.0) {
			s = p + t*d;
			if(abs(s.y) <= 1.0 && abs(s.x) <= 1.0) {
				p = s;
				return true;
			}
		}
	}
	return false;
}

float escapeCubeDistance(vec3 p, vec3 d, vec3 invD) {
	// Determine minimum distance to a cube boundary from given position and direction.
	// Add an additional scaling factor to escape the current cube
	vec3 s = (unitEpsilon*sign(d) - p)*invD;
	return min(min(s.x, s.y), s.z);
}

uint voxelIndex(inout vec3 p, inout float scale, int maxDepth) {
	uint index = 0;

	// Determine smallest scale voxel cell this point exists in
	int i = 0;
	do {
		if(index == emptyVoxel) return index;

		scale *= 0.5;
		if(p.x > 0.0) {
			if(p.y > 0.0) {
				if(p.z > 0.0) {
					index = voxelOctree.voxels[index].btr;
					p -= btrCell;
				} else {
					index = voxelOctree.voxels[index].ftr;
					p -= ftrCell;
				}
			} else {
				if(p.z > 0.0) {
					index = voxelOctree.voxels[index].bbr;
					p -= bbrCell;
				} else {
					index = voxelOctree.voxels[index].fbr;
					p -= fbrCell;
				}
			}
		} else {
			if(p.y > 0.0) {
				if(p.z > 0.0) {
					index = voxelOctree.voxels[index].btl;
					p -= btlCell;
				} else {
					index = voxelOctree.voxels[index].ftl;
					p -= ftlCell;
				}
			} else {
				if(p.z > 0.0) {
					index = voxelOctree.voxels[index].bbl;
					p -= bblCell;
				} else {
					index = voxelOctree.voxels[index].fbl;
					p -= fblCell;
				}
			}
		}
		p += p;
	} while(++i < maxDepth && voxelOctree.voxels[index].vtype == 0);
	return index;
}

float goalVoxelTraversal(inout vec3 p, vec3 d) {
	float pd = dot(d, p);
	float r = 4.0*(pd*pd - dot(p, p) + goalRadiusSquared);
	if(r >= 0.0) {
		float t = (-2.0*pd - sqrt(r))/2.0;
		vec3 s = t * d + p;
		p = s;
		return t;
	}
	return -1.0;
}

float castShadowRay(vec3 p, vec3 d, vec3 invD, int maxDepth) {
	if(!projectToRootVoxel(p, d, invD)) return 1.0;

	int i = 0;
	do {
		vec3 s = p;
		float scale = 1.0;
		uint index = voxelIndex(s, scale, maxDepth);

		// Is empty or filled?
		if(index == emptyVoxel) {
			float t = escapeCubeDistance(s, d, invD) * scale;
			p += t * d;
		} else {
			Voxel voxel = voxelOctree.voxels[index];
			if(voxel.vtype == 2) {
				// We are in a goal voxel! Traverse and check for hit
				float t = goalVoxelTraversal(s, d);
				if(t >= 0.0) {
					return 0.0;
				} else {
					// Copy pasta empty cell
					t = escapeCubeDistance(s, d, invD) * scale;
					p += t * d;
				}
			} else {
				return 0.0;
			}
		}
		if(!insideCube(p)) return 1.0;
	} while(++i < maxIterations);
	return 0.0;
}

const float maxBrightness = 1.3;
const float maxBrightnessR2 = maxBrightness*maxBrightness;
vec4 scaleColor(float si, vec4 col) {
	float temp = 1.0 - si/maxIterationsF;
	return mix(fogColour, col, temp);
}

vec3 gradient;
vec3 phongLighting(vec3 c, float shadow) {
	vec3 diffuse = max(dot(gradient, push.light_dir), 0.0) * lightColor;
	return (ambientLight + diffuse * shadow) * c;
}

vec4 escapeColour(vec3 d) {
	float temp = dot(dirY, d);
	vec4 groundSkyColour = mix(groundColour, skyColour, (sqrt(abs(temp))*sign(temp) + 1.0)/2.0);
	return mix(groundSkyColour, vec4(lightColor, 1.0), clamp(64.0*dot(d, push.light_dir) - 63.0, 0.0, 1.0));
}

// The MIT License
// Copyright © 2013 Inigo Quilez
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
// https://www.youtube.com/c/InigoQuilez
// https://iquilezles.org
// https://www.shadertoy.com/view/XdXGW8
vec3 iq_inspired_position_noise(ivec3 z) {
    // 2D to 1D  (feel free to replace by some other)
    int n = z.x + z.y*1111 + z.z*5227;

    // Hugo Elias hash (feel free to replace by another one)
    n = (n<<13)^n;
    n = (n*(n*n*15731+789221)+1376312589)>>16;

    // simple random vectors
    return vec3(cos(float(n)), sin(float(n)), 2.0*fract(log(abs(float(n)) + 1.0)) - 1.0);
}
float iq_inspired_noise(vec3 p) {
    ivec3 i = ivec3(floor(p));
	vec3 f = fract(p);

	vec3 u = f*f*(3.0 - 2.0*f); // feel free to replace by a quintic smoothstep instead

    return mix(
		mix(mix(dot(iq_inspired_position_noise(i), f),
				dot(iq_inspired_position_noise(i + ivec3(1, 0, 0)), f - vec3(1.0, 0.0, 0.0)), u.x),
			mix(dot(iq_inspired_position_noise(i + ivec3(0, 1, 0)), f - vec3(0.0, 1.0, 0.0)),
				dot(iq_inspired_position_noise(i + ivec3(1, 1, 0)), f - vec3(1.0, 1.0, 0.0)), u.x), u.y),
		mix(mix(dot(iq_inspired_position_noise(i + ivec3(0, 0, 1)), f - vec3(0.0, 0.0, 1.0)),
				dot(iq_inspired_position_noise(i + ivec3(1, 0, 1)), f - vec3(1.0, 0.0, 1.0)), u.x),
			mix(dot(iq_inspired_position_noise(i + ivec3(0, 1, 1)), f - vec3(0.0, 1.0, 1.0)),
				dot(iq_inspired_position_noise(i + ivec3(1, 1, 1)), f - vec3(1.0, 1.0, 1.0)), u.x), u.y), u.z);
}

const float minTravel = 0.000005;
vec4 castVoxelRay(vec3 p, vec3 d) {
	// Remember source position of cast
	vec3 origin = p;

	// Reset gradient
	gradient = vec3(0.0);

	// Move some non-zero distance
	p += minTravel * d;
	vec3 invD = 1.0 / d;
	if(!projectToRootVoxel(p, d, invD)) return escapeColour(d);

	vec4 col = vec4(0.0);
	int reflections = 0;

	int i = 0;
	do {
		vec3 s = p;
		float scale = 1.0;
		int maxDepth = clamp(int(9.85 - 1.4427*log(length(p - origin))), 3, globalMaxDepth);
		uint index = voxelIndex(s, scale, maxDepth);

		// Is empty or filled?
		if(index == emptyVoxel) {
			float t = escapeCubeDistance(s, d, invD) * scale;
			p += t * d;
		} else {
			Voxel voxel = voxelOctree.voxels[index];
			if(voxel.vtype == 2) {
				// We are in a goal voxel! Traverse and check for hit
				float t = goalVoxelTraversal(s, d);
				float r2 = dot(s, s);
				if(t >= 0.0 || r2 < goalRadiusSquared) {
					t *= scale;
					p += t * d;
					gradient = normalize(s);

					float colTemp = sin(7.0*push.time + 1.25*s.x + 1.5*s.y - 1.5*s.z);
					float colTemp2 = cos(8.0*push.time - 2.5*s.x * 3.0*s.y * 2.0*s.z);
					colTemp = (colTemp + colTemp2) / 2.0;
					vec3 portalCol = voxel.averageColour.xyz;
					portalCol = mix(portalCol, vec3(0.0), min(colTemp, tan(8.0*push.time - 12.0*(dot(s, d)))));

					col += col + col + col + vec4(phongLighting(portalCol, castShadowRay(p, push.light_dir, 1.0 / push.light_dir, maxDepth)), 1.0);

					return scaleColor(i, col/col.w);
				} else {
					// Gravity-based raytracing
					vec3 c = p - scale*s;
					for(int j = 0; j < maxIterations; ++j) {
						r2 = dot(s, s);
						if(!insideCube(s) || r2 < goalRadiusSquared) {
							break;
						}
						d += 0.024 * s / (r2*sqrt(r2));
						d = normalize(d);
						s += d * 0.075;
					}
					vec3 q = c + scale*s;
					invD = 1.0 / d;
					p = q;
				}
			} else {
				// We have hit a mirror voxel. Reflect and continue
				gradient = cubeNorm(s);
				p += projectToOutsideDistance(s) * scale;

				if(voxel.vtype == 3 && reflections < 2) {
					reflections += 1;
					d -= 2.0*dot(d, gradient)*gradient;
					invD = 1.0 / d;

					vec3 uv = 32.0*s;
					mat2 m = mat2(1.6,  1.2, -1.2,  1.6);
					float mirrorFuzz = 0.5*iq_inspired_noise(uv); uv.xy = m*uv.xy;
					mirrorFuzz += 0.25*iq_inspired_noise(uv); uv.yz = -(m*uv.yz);
					mirrorFuzz += 0.125*iq_inspired_noise(uv); uv.zx = m*uv.zx;
					mirrorFuzz += 0.0625*iq_inspired_noise(uv); uv.yx = -(m*uv.yx);
					mirrorFuzz += 0.03125*iq_inspired_noise(uv); uv.zy = m*uv.zy;
					mirrorFuzz += 0.015625*iq_inspired_noise(uv); uv.xz = -(m*uv.xz);
					mirrorFuzz += 0.0078125*iq_inspired_noise(uv);
					mirrorFuzz *= 0.16;

					col += col + col + col + vec4(vec3(mirrorFuzz) + phongLighting(voxel.averageColour.xyz, castShadowRay(p, push.light_dir, 1.0 / push.light_dir, maxDepth)), 1.0);
				} else {
					vec3 t = abs(s);
					float borderOutline = pow(min(abs(t.x - t.y), min(abs(t.x - t.z), abs(t.y - t.z))), 0.16);
					col += col + col + col + vec4(borderOutline*phongLighting(voxel.averageColour.xyz, castShadowRay(p, push.light_dir, 1.0 / push.light_dir, maxDepth)), 1.0);

					return scaleColor(i, col/col.w);
				}
			}
		}
	} while(++i < maxIterations && insideCube(p));
	col += col + col + col + escapeColour(d);
	return scaleColor(i, col/col.w);
}

const float fov = (pi/1.75) / 2.0;
const float fovY = sin(fov);
float fovX = push.aspect_ratio * fovY;
void main(void) {
	vec3 direction = normalize(vec3(coord.x*fovX, -coord.y*fovY, 1.0));
	direction = rotateByQuaternion(direction, push.camera_quaternion);
	vec3 pos = push.camera_position;

	fragColor = castVoxelRay(pos, direction);
}