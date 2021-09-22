include_directories(.)

# library
add_subdirectory(tinypt)

# examples
option(TINYPT_ENABLE_EXAMPLES "" ON)
if (TINYPT_ENABLE_EXAMPLES)
    add_subdirectory(examples)
endif ()
