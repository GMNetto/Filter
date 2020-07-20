project(point-cloud)

set(app_SOURCES
	src/main.cpp
)

add_executable(pf_pos_normal ${app_SOURCES})
target_link_libraries(pf_pos_normal PUBLIC lib)