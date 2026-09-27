#include "earth_sgp4.h"

#include "CoordGeodetic.h"
#include "DateTime.h"
#include "SGP4.h"
#include "Tle.h"

#include <cctype>
#include <cmath>
#include <cstdint>
#include <exception>
#include <fstream>
#include <memory>
#include <new>
#include <string>
#include <utility>

struct earth_sgp4 {
    std::unique_ptr<libsgp4::SGP4> propagator;
    std::string last_error;
};

namespace {

constexpr int64_t kUnixEpochTicks = 62135596800000000LL;
constexpr int64_t kTicksPerSecond = 1000000LL;
constexpr int64_t kMinimumUnixSecond = -62135596800LL;
constexpr int64_t kMaximumUnixSecond = 253402300799LL;
constexpr const char kNullHandleError[] = "earth_sgp4 handle is null";

int SetError(earth_sgp4_t *handle, int status, const char *message) noexcept
{
    if (handle != nullptr) {
        try {
            handle->last_error = message != nullptr ? message : "Unknown SGP4 error";
        } catch (...) {
            // Avoid allowing allocation failures to escape the C ABI.
            handle->last_error.clear();
        }
    }
    return status;
}

void ClearError(earth_sgp4_t *handle) noexcept
{
    if (handle != nullptr) {
        handle->last_error.clear();
    }
}

std::string TrimAsciiWhitespace(const std::string &value)
{
    size_t first = 0;
    while (first < value.size()
           && std::isspace(static_cast<unsigned char>(value[first])) != 0) {
        ++first;
    }

    size_t last = value.size();
    while (last > first
           && std::isspace(static_cast<unsigned char>(value[last - 1])) != 0) {
        --last;
    }
    return value.substr(first, last - first);
}

bool IsTleLine(const std::string &line, char line_number)
{
    return line.size() >= 2 && line[0] == line_number && line[1] == ' ';
}

int LoadTle(
    earth_sgp4_t *handle,
    const char *name,
    const char *line_one,
    const char *line_two) noexcept
{
    try {
        const libsgp4::Tle tle(
            name != nullptr ? name : "",
            line_one,
            line_two);
        auto propagator = std::make_unique<libsgp4::SGP4>(tle);
        handle->propagator = std::move(propagator);
        ClearError(handle);
        return EARTH_SGP4_SUCCESS;
    } catch (const std::exception &error) {
        return SetError(handle, EARTH_SGP4_ERROR_TLE, error.what());
    } catch (...) {
        return SetError(handle, EARTH_SGP4_ERROR_INTERNAL, "Unknown TLE load failure");
    }
}

} // namespace

extern "C" earth_sgp4_t *earth_sgp4_create(void)
{
    try {
        return new earth_sgp4_t();
    } catch (...) {
        return nullptr;
    }
}

extern "C" void earth_sgp4_destroy(earth_sgp4_t *handle)
{
    delete handle;
}

extern "C" int earth_sgp4_load_tle(
    earth_sgp4_t *handle,
    const char *name,
    const char *line_one,
    const char *line_two)
{
    if (handle == nullptr || line_one == nullptr || line_two == nullptr) {
        return SetError(handle, EARTH_SGP4_ERROR_INVALID_ARGUMENT,
                        "TLE handle and both TLE lines are required");
    }
    return LoadTle(handle, name, line_one, line_two);
}

extern "C" int earth_sgp4_load_tle_file(earth_sgp4_t *handle, const char *path)
{
    if (handle == nullptr || path == nullptr) {
        return SetError(handle, EARTH_SGP4_ERROR_INVALID_ARGUMENT,
                        "TLE handle and path are required");
    }

    try {
        std::ifstream input(path);
        if (!input.is_open()) {
            return SetError(handle, EARTH_SGP4_ERROR_IO, "Could not open TLE file");
        }

        std::string candidate_name;
        std::string line_one;
        std::string line_two;
        std::string line;
        while (std::getline(input, line)) {
            const std::string trimmed = TrimAsciiWhitespace(line);
            if (trimmed.empty()) {
                continue;
            }
            if (IsTleLine(trimmed, '1')) {
                line_one = trimmed;
                line_two.clear();
                continue;
            }
            if (IsTleLine(trimmed, '2')) {
                if (!line_one.empty()) {
                    line_two = trimmed;
                    break;
                }
                continue;
            }
            if (line_one.empty()) {
                candidate_name = trimmed;
            }
        }

        if (line_one.empty() || line_two.empty()) {
            return SetError(handle, EARTH_SGP4_ERROR_TLE,
                            "TLE file does not contain a complete TLE pair");
        }
        return LoadTle(handle, candidate_name.c_str(), line_one.c_str(), line_two.c_str());
    } catch (const std::exception &error) {
        return SetError(handle, EARTH_SGP4_ERROR_IO, error.what());
    } catch (...) {
        return SetError(handle, EARTH_SGP4_ERROR_INTERNAL, "Unknown TLE file load failure");
    }
}

extern "C" int earth_sgp4_propagate_unix_utc(
    earth_sgp4_t *handle,
    int64_t unix_seconds,
    int32_t microseconds,
    double *latitude_radians,
    double *longitude_radians,
    double *altitude_kilometres)
{
    if (handle == nullptr || latitude_radians == nullptr || longitude_radians == nullptr
        || altitude_kilometres == nullptr) {
        return SetError(handle, EARTH_SGP4_ERROR_INVALID_ARGUMENT,
                        "Handle and all geodetic output pointers are required");
    }
    if (handle->propagator == nullptr) {
        return SetError(handle, EARTH_SGP4_ERROR_NOT_LOADED, "No TLE is loaded");
    }
    if (microseconds < 0 || microseconds >= kTicksPerSecond
        || unix_seconds < kMinimumUnixSecond || unix_seconds > kMaximumUnixSecond) {
        return SetError(handle, EARTH_SGP4_ERROR_TIME_RANGE,
                        "UTC timestamp is outside the supported DateTime range");
    }

    try {
        const int64_t ticks = kUnixEpochTicks + unix_seconds * kTicksPerSecond + microseconds;
        const libsgp4::DateTime utc(ticks);
        const libsgp4::CoordGeodetic position =
            handle->propagator->FindPosition(utc).ToGeodetic();

        if (!std::isfinite(position.latitude) || !std::isfinite(position.longitude)
            || !std::isfinite(position.altitude)) {
            return SetError(handle, EARTH_SGP4_ERROR_PROPAGATION,
                            "SGP4 returned a non-finite geodetic position");
        }

        *latitude_radians = position.latitude;
        *longitude_radians = position.longitude;
        *altitude_kilometres = position.altitude;
        ClearError(handle);
        return EARTH_SGP4_SUCCESS;
    } catch (const std::exception &error) {
        return SetError(handle, EARTH_SGP4_ERROR_PROPAGATION, error.what());
    } catch (...) {
        return SetError(handle, EARTH_SGP4_ERROR_INTERNAL, "Unknown SGP4 propagation failure");
    }
}

extern "C" const char *earth_sgp4_last_error(const earth_sgp4_t *handle)
{
    return handle != nullptr ? handle->last_error.c_str() : kNullHandleError;
}

extern "C" void earth_sgp4_clear_error(earth_sgp4_t *handle)
{
    ClearError(handle);
}
