#include <Kokkos_Core.hpp>
#include <aether/physics/api.hpp>
#include <aether/core/config.hpp>
#include <aether/io/binary_snapshot.hpp>
#include <aether/core/config_build.hpp>
#include <aether/io/snapshot.hpp>
#include <aether/io/plaintext_snapshot.hpp>
#include <aether/io/metadata.hpp>
#include <stdexcept>

namespace aether::io {

void write_snapshot(aether::core::Simulation& sim, snapshot_request& req) {
    sim.time.write_num++;
    for (output_format type : req.formats) {
        switch (type) {
            case output_format::plain_txt:
                write_plaintext_snapshot(sim, req);
                break;

            case output_format::binary:
                write_binary_snapshot(sim,req);
                break;

            default:
                throw std::runtime_error("write_snapshot: unknown output format");
        }
    }
}
} // namespace aether::io
