project(point-cloud)

set(app_SOURCES
	src/main.cpp
)

add_executable(temporal_filter ${app_SOURCES})
target_link_libraries(temporal_filter PUBLIC lib)