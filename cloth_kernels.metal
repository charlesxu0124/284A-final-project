#include <metal_stdlib>
using namespace metal;

inline float3 table_contact(float3 previous, float3 proposed, float friction) {
    const float penetration = max(-proposed.z, 0.0f);
    const float2 slip = proposed.xy - previous.xy;
    const float fraction = min(friction * penetration / max(length(slip), 1e-12f), 1.0f);
    return float3(proposed.xy - slip * fraction, proposed.z + penetration);
}

// One threadgroup per cloth. Each thread accumulates the springs at one vertex.
kernel void spring_step(
    const device packed_float3* positions,
    const device packed_float3* velocities,
    device packed_float3* next_positions,
    device packed_float3* next_velocities,
    const device packed_float3* targets,
    const device int* slots,
    const device int* offsets,
    const device int* neighbors,
    const device float2* material,
    const device int* config,
    const device float* physical,
    constant float& h,
    constant float& damping,
    constant uint& held_count,
    uint vertex_id [[thread_index_in_threadgroup]],
    uint environment [[threadgroup_position_in_grid]]) {
    const uint vertex_count = config[0];
    const uint substeps = config[1];
    const bool active = vertex_id < vertex_count;
    const uint index = environment * vertex_count + vertex_id;
    const int slot = active ? slots[vertex_id] : -1;
    const bool pinned = slot >= 0;
    const float3 initial = active ? float3(positions[index]) : float3(0.0f);
    const float3 travel = pinned ? float3(targets[environment * held_count + slot]) - initial : float3(0.0f);
    float3 x = initial;
    float3 velocity = active ? float3(velocities[index]) : float3(0.0f);
    threadgroup float3 shared_positions[VERTEX_CAPACITY];

    for (uint step = 1; step <= substeps; ++step) {
        if (active) {
            shared_positions[vertex_id] = x;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active) {
            float3 proposed;
            if (pinned) {
                proposed = initial + travel * (float(step) / float(substeps));
            } else {
                float3 force(0.0f);
                for (int edge = offsets[vertex_id]; edge < offsets[vertex_id + 1]; ++edge) {
                    const float3 displacement = shared_positions[neighbors[edge]] - x;
                    const float distance = length(displacement);
                    const float2 spring = material[edge];
                    force += displacement * (spring.y * (distance - spring.x) / max(distance, 1e-12f));
                }
                const float3 predicted_velocity = velocity * damping + float3(0.0f, 0.0f, -9.81f * h)
                    + force * physical[0] * h;
                proposed = table_contact(x, x + predicted_velocity * h, physical[1]);
            }
            velocity = (proposed - x) / h;
            x = proposed;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (active) {
        next_positions[index] = packed_float3(x);
        next_velocities[index] = packed_float3(velocity);
    }
}

// Edges in one color have disjoint endpoints. The barrier preserves Gauss–Seidel order.
kernel void xpbd_step(
    const device packed_float3* positions,
    const device packed_float3* velocities,
    device packed_float3* next_positions,
    device packed_float3* next_velocities,
    const device packed_float3* targets,
    const device int* slots,
    const device int* offsets,
    const device int2* endpoints,
    const device float2* material,
    const device int* config,
    const device float* physical,
    constant float& h,
    constant float& damping,
    constant uint& held_count,
    uint vertex_id [[thread_index_in_threadgroup]],
    uint environment [[threadgroup_position_in_grid]]) {
    const uint vertex_count = config[0];
    const uint substeps = config[1];
    const uint sweeps = config[2];
    const uint colors = config[3];
    const bool active = vertex_id < vertex_count;
    const uint index = environment * vertex_count + vertex_id;
    const int slot = active ? slots[vertex_id] : -1;
    const bool pinned = slot >= 0;
    const float3 initial = active ? float3(positions[index]) : float3(0.0f);
    const float3 travel = pinned ? float3(targets[environment * held_count + slot]) - initial : float3(0.0f);
    float3 x = initial;
    float3 velocity = active ? float3(velocities[index]) : float3(0.0f);
    threadgroup float3 shared_positions[VERTEX_CAPACITY];

    for (uint step = 1; step <= substeps; ++step) {
        float multipliers[32] = {};
        if (active) {
            shared_positions[vertex_id] = pinned ? initial + travel * (float(step) / float(substeps))
                : x + (velocity * damping + float3(0.0f, 0.0f, -9.81f * h)) * h;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint sweep = 0; sweep < sweeps; ++sweep) {
            for (uint color = 0; color < colors; ++color) {
                const uint edge = offsets[color] + vertex_id;
                if (edge < uint(offsets[color + 1])) {
                    const int2 pair = endpoints[edge];
                    const float w_left = slots[pair.x] < 0 ? physical[0] : 0.0f;
                    const float w_right = slots[pair.y] < 0 ? physical[0] : 0.0f;
                    const float3 displacement = shared_positions[pair.y] - shared_positions[pair.x];
                    const float distance = length(displacement);
                    const float2 constraint = material[edge];
                    const float alpha = constraint.y / (h * h);
                    const float change = (constraint.x - distance - alpha * multipliers[color])
                        / (w_left + w_right + alpha);
                    multipliers[color] += change;
                    const float3 correction = displacement * (change / max(distance, 1e-12f));
                    shared_positions[pair.x] -= w_left * correction;
                    shared_positions[pair.y] += w_right * correction;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

        if (active) {
            float3 proposed = shared_positions[vertex_id];
            if (!pinned) {
                proposed = table_contact(x, proposed, physical[1]);
            }
            velocity = (proposed - x) / h;
            x = proposed;
        }
    }

    if (active) {
        next_positions[index] = packed_float3(x);
        next_velocities[index] = packed_float3(velocity);
    }
}
