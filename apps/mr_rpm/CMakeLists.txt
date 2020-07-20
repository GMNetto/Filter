project(point-cloud)

set(app_SOURCES
	src/main.cpp
)

add_executable(mr_rpm ${app_SOURCES})
target_link_libraries(mr_rpm PUBLIC lib)