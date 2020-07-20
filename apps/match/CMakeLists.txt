
project(point-cloud)

set(app_SOURCES
	src/main.cpp
)

add_executable(match ${app_SOURCES})
target_link_libraries(match PUBLIC lib)