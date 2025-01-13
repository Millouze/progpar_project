__kernel void computeBodiesAcceleration(
    __global const float *in_qx,
    __global const float *in_qy,
    __global const float *in_qz,
    __global const float *in_m,
    __global float *out_ax,
    __global float *out_ay,
    __global float *out_az,
    const unsigned long n,
    const float soft_squared,
    const float g
)
{
    unsigned long i = get_global_id(0);
    if (i > n)
        return;

    float ax = 0;
    float ay = 0;
    float az = 0;
    for (unsigned long j = 0; j < n; j++) {
        const float rijx = in_qx[j] - in_qx[i];
        const float rijy = in_qy[j] - in_qy[i];
        const float rijz = in_qz[j] - in_qz[i];

        // compute the || rij ||² distance between body i and body j
        const float rij_squared = rijx * rijx + rijy * rijy + rijz * rijz;

        const float inv_s = rsqrt(rij_squared + soft_squared);

        
        // compute the acceleration value between body i and body j: || ai || = G.mj / (|| rij ||² + e²)^{3/2}
        const float ai = in_m[j] * inv_s * inv_s * inv_s * g;

        // add the acceleration value into the acceleration vector: ai += || ai ||.rij
        ax += ai * rijx;
        ay += ai * rijy;
        az += ai * rijz;
    }

    out_ax[i] += ax ;
    out_ay[i] += ay ;
    out_az[i] += az ;
}