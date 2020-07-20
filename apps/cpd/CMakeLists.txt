
project(point-cloud)

set(app_SOURCES
	src/main.cpp
)

add_executable(cpd ${app_SOURCES})
target_link_libraries(cpd PUBLIC lib)