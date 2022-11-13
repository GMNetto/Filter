
project(point-cloud)

set(app_SOURCES
	src/main.cpp
)

add_executable(filter ${app_SOURCES})
target_link_libraries(filter PUBLIC lib)
