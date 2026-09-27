/*
 * Minimal C ABI around the vendored libsgp4 implementation.
 *
 * Latitude and longitude results are radians. Altitude results are kilometres.
 */
#ifndef EARTH_NATIVE_EARTH_SGP4_H
#define EARTH_NATIVE_EARTH_SGP4_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct earth_sgp4 earth_sgp4_t;

enum earth_sgp4_status {
    EARTH_SGP4_SUCCESS = 0,
    EARTH_SGP4_ERROR_INVALID_ARGUMENT = -1,
    EARTH_SGP4_ERROR_IO = -2,
    EARTH_SGP4_ERROR_TLE = -3,
    EARTH_SGP4_ERROR_NOT_LOADED = -4,
    EARTH_SGP4_ERROR_TIME_RANGE = -5,
    EARTH_SGP4_ERROR_PROPAGATION = -6,
    EARTH_SGP4_ERROR_INTERNAL = -7,
};

/* Allocates an empty propagator handle, or returns NULL on allocation failure. */
earth_sgp4_t *earth_sgp4_create(void);

/* Releases a handle. Passing NULL is allowed. */
void earth_sgp4_destroy(earth_sgp4_t *handle);

/*
 * Parses and loads an exact two-line element set. `name` may be NULL; both TLE
 * lines must be non-NULL and retain their standard 69-character formatting.
 * A failed load leaves an already-loaded orbit usable.
 */
int earth_sgp4_load_tle(
    earth_sgp4_t *handle,
    const char *name,
    const char *line_one,
    const char *line_two);

/*
 * Loads the first complete TLE pair in a text file. A preceding non-TLE line
 * is used as the optional satellite name. Leading/trailing file whitespace is
 * discarded before the two TLE lines are parsed.
 */
int earth_sgp4_load_tle_file(earth_sgp4_t *handle, const char *path);

/*
 * Propagates a loaded TLE at a POSIX UTC timestamp. `microseconds` must be in
 * [0, 999999]. Results use the vendored library's geodetic convention:
 * latitude and longitude are radians and altitude is kilometres.
 */
int earth_sgp4_propagate_unix_utc(
    earth_sgp4_t *handle,
    int64_t unix_seconds,
    int32_t microseconds,
    double *latitude_radians,
    double *longitude_radians,
    double *altitude_kilometres);

/* Returns a non-owning diagnostic string valid until the next call on handle. */
const char *earth_sgp4_last_error(const earth_sgp4_t *handle);

/* Clears the diagnostic string. Passing NULL is allowed. */
void earth_sgp4_clear_error(earth_sgp4_t *handle);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* EARTH_NATIVE_EARTH_SGP4_H */
