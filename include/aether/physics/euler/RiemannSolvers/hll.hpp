#pragma once 
#include <aether/core/config.hpp>
#include <aether/core/enums.hpp>
#include <aether/physics/euler/variable_structs.hpp>
#include <cmath>

namespace aether::physics::euler{
    AETHER_INLINE prims hll(prims &L, prims &R, const double gamma){
        prims Flux;
        const double aL2 = L.p * gamma /L.rho;
        const double aR2 = R.p * gamma /R.rho;

        const double aL = std::sqrt(aL2);
        const double aR = std::sqrt(aR2);

        const double SL = std::fmin(L.vx - aL, R.vx - aR);
        const double SR = std::fmax(L.vx + aL, R.vx + aR);

        if (0.0 <= SL){
            const double v2 = L.vx*L.vx + L.vy*L.vy + L.vz*L.vz;
            double E = L.p/(gamma-1.0) + 0.5*L.rho*v2;

            Flux.rho = L.rho * L.vx;
            Flux.vx = L.rho * L.vx * L.vx + L.p;
            Flux.vy = L.rho * L.vx * L.vy; 
            Flux.vz = L.rho * L.vx * L.vz; 
            Flux.p = L.vx*(E + L.p);
        }else if (0 >= SR){
            const double v2 = R.vx*R.vx + R.vy*R.vy + R.vz*R.vz;
            double E = R.p/(gamma-1.0) + 0.5*R.rho*v2;

            Flux.rho = R.rho * R.vx;
            Flux.vx = R.rho * R.vx * R.vx + R.p;
            Flux.vy = R.rho * R.vx * R.vy; 
            Flux.vz = R.rho * R.vx * R.vz; 
            Flux.p = R.vx*(E + R.p);
        }else {
            double FL[5], FR[5], UL[5], UR[5], F[5];

            const double v2L = L.vx*L.vx + L.vy*L.vy + L.vz*L.vz;
            double EL = L.p/(gamma-1.0) + 0.5*L.rho*v2L;

            FL[0] = L.rho * L.vx;
            FL[1] = L.rho * L.vx * L.vx + L.p;
            FL[2] = L.rho * L.vx * L.vy; 
            FL[3] = L.rho * L.vx * L.vz; 
            FL[4] = L.vx*(EL + L.p);

            const double v2R = R.vx*R.vx + R.vy*R.vy + R.vz*R.vz;
            double ER = R.p/(gamma-1.0) + 0.5*R.rho*v2R;

            FR[0] = R.rho * R.vx;
            FR[1] = R.rho * R.vx * R.vx + R.p;
            FR[2] = R.rho * R.vx * R.vy; 
            FR[3] = R.rho * R.vx * R.vz; 
            FR[4] = R.vx*(ER + R.p);        
            
            UL[0] = L.rho;
            UL[1] = L.vx*L.rho;
            UL[2] = L.vy*L.rho;
            UL[3] = L.vz*L.rho;
            UL[4] = L.p/(gamma-1.0) + 0.5*L.rho*v2L;

            UR[0] = R.rho;
            UR[1] = R.vx*R.rho;
            UR[2] = R.vy*R.rho;
            UR[3] = R.vz*R.rho;
            UR[4] = R.p/(gamma-1.0) + 0.5*R.rho*v2R;

            const double diff = 1.0/(SR-SL);

            for (int i = 0; i < 5; ++i){
                F[i] = diff*(SR*FL[i] - SL*FR[i] + SR*SL*(UR[i] - UL[i]));
            }

            Flux.rho = F[0];
            Flux.vx = F[1];
            Flux.vy = F[2];
            Flux.vz = F[3];
            Flux.p = F[4];
        }
        return Flux;
    }
}