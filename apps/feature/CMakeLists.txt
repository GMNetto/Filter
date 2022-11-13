
project(point-cloud)

set(app_SOURCES
	src/main.cpp
)

add_executable(feature ${app_SOURCES})
target_link_libraries(feature PUBLIC lib)
