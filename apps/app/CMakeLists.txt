
project(point-cloud)

set(app_SOURCES
	src/main.cpp
)

add_executable(app ${app_SOURCES})
target_link_libraries(app PUBLIC lib)