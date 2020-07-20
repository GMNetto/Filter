#include "pfpoint.hpp"
#include "util.hpp"

#include <pcl/octree/octree_search.h>
#include <pcl/common/common_headers.h>
#include <pcl/filters/bilateral.h>
#include <pcl/filters/fast_bilateral.h>
#include <cmath>
#include <unordered_map>

template <typename Point, typename OctreePoint>
void copy_n_point(typename pcl::PointCloud<Point>::ConstPtr cloud, std::deque<int>& neighbors, typename pcl::octree::OctreePointCloudSearch<OctreePoint>& octree
    ,std::unordered_map<int, std::vector<int>>& visited, int current_elem, float radius=1.1) {
    std::vector<int> pointIdxRadiusSearch;
    std::vector<float> pointRadiusSquaredDistance;

    OctreePoint point;
    point.x = cloud->points[current_elem].x;
    point.y = cloud->points[current_elem].y;
    point.z = cloud->points[current_elem].z;
    if (octree.nearestKSearch (point, 10, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0)
    {
        for (size_t j = 0; j < pointIdxRadiusSearch.size (); ++j) {


            if (pointIdxRadiusSearch[j] != current_elem) {
                //std::cout << "Inserting into N: " << pointIdxRadiusSearch[j] << std::endl;
                neighbors.push_back(pointIdxRadiusSearch[j]);
            }
        }
    }
    visited.emplace(current_elem, pointIdxRadiusSearch);
}

template <typename Point, typename Normal>
float permeability_radial_point(Point& p, Normal& pn, Point& n, Normal& nn, float sigma_s, float sigma_r, float sigma_d) {
    float cos_val = dot_product_normal<Normal, Normal>(pn, nn);
    float fdist = acos(std::min(1.0f, cos_val));

    float denominator = M_PI*sigma_r;
    float aux = pow(fdist/denominator, sigma_s);

    pcl::PointXYZ diff(p.x - n.x, p.y - n.y, p.z - n.z);

    float ddist = norm<pcl::PointXYZ>(diff);
    return std::min(1/(1 + aux), 1.0f);
}

void filter_PF_point(int index, int neighbor_idx, pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr cloud,
        std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr>& ls, float sigma_s, float sigma_r, float sigma_d) {

    pcl::PointXYZRGBNormal current = cloud->points[index];

    pcl::PointXYZRGBNormal current_normal = cloud->points[index], current_point = cloud->points[index];
    pcl::PointXYZRGBNormal neighbor_point = cloud->points[neighbor_idx];
    pcl::PointXYZRGBNormal neighbor_normal = cloud->points[neighbor_idx];


    float permeability = permeability_radial_point<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>(current_point, current_normal, neighbor_point, neighbor_normal, sigma_s, sigma_r, sigma_d);
    float sq_per = permeability*permeability;

    pcl::PointXYZRGBNormal diff;
    diff.getArray3fMap() = current_point.getArray3fMap() - neighbor_point.getArray3fMap();
    float proj_current = dot_product<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>(diff, neighbor_normal);

    pcl::PointXYZRGBNormal aux;
    sum_multi_scalar2<pcl::PointXYZRGBNormal>(ls[0]->points[index], sq_per, proj_current, aux);

    ls[0]->points[neighbor_idx].getNormalVector3fMap() += aux.getNormalVector3fMap();
    
    sum_multi_scalar<pcl::PointXYZRGBNormal>(ls[1]->points[neighbor_idx], 1, permeability, ls[1]->points[neighbor_idx]);
    
    pcl::Normal one(1.f,1.f,1.f);
    sum_multi_scalar2<pcl::PointXYZRGBNormal>(ls[2]->points[index], sq_per, 1, aux);
    ls[2]->points[neighbor_idx].getNormalVector3fMap() += aux.getNormalVector3fMap();

    sum_multi_scalar<pcl::PointXYZRGBNormal>(ls[3]->points[neighbor_idx], 1, permeability, ls[3]->points[neighbor_idx]);
}

template <typename ValVisited>
void filter_PF_neighbors_point(int index, pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr cloud,
    std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr>& ls, typename std::unordered_map<int, ValVisited>& visited,
    const std::vector<int>& pointIdxRadiusSearch, float sigma_s, float sigma_r, float sigma_d) {

    divide_normal<pcl::PointXYZRGBNormal>(ls[0]->points[index], ls[1]->points[index], ls[0]->points[index]);
    divide_normal<pcl::PointXYZRGBNormal>(ls[2]->points[index], ls[3]->points[index], ls[2]->points[index]);

    for (size_t j = 0; j < pointIdxRadiusSearch.size (); ++j) {
        int neighbor_idx = pointIdxRadiusSearch[j];
        if ((neighbor_idx != index)  && visited.find(neighbor_idx) == visited.end()) {
            filter_PF_point(index, neighbor_idx, cloud, ls, sigma_s, sigma_r, sigma_d);
        }
    }    
}

void BFS_point(pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr cloud, std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr>& results, std::vector<int>& order_visit,
        pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>& octree, float sigma_s, float sigma_r, float sigma_d) {

    std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> ls(4);
    for (int i=0; i < 3; i+=2) {
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr l (new pcl::PointCloud<pcl::PointXYZRGBNormal>());
        ls[i] = l;
        pcl::copyPointCloud(*cloud, *ls[i]);
        initialize<pcl::PointXYZRGBNormal>(ls[i], 0);
    }

    for (int i=1; i < 4; i+=2) {
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr l (new pcl::PointCloud<pcl::PointXYZRGBNormal>());
        ls[i] = l;
        pcl::copyPointCloud(*cloud, *ls[i]);
        initialize<pcl::PointXYZRGBNormal>(ls[i], 0.000001);
    }
    std::unordered_map<int, std::vector<int>> visited;

    std::deque<int> neighbors;
    order_visit.push_back(0);
    copy_n_point<pcl::PointXYZRGBNormal, pcl::PointXYZ>(cloud, neighbors, octree, visited, 0);
    filter_PF_neighbors_point<std::vector<int>>(0, cloud, ls, visited, visited[0], sigma_s, sigma_r, sigma_d);
    int counter = 0;
    while (neighbors.size() != 0 && counter < 50000) {
        int next_elem = neighbors.front();
        neighbors.pop_front();
        if (visited.find(next_elem) == visited.end()) {
            order_visit.push_back(next_elem);
            copy_n_point<pcl::PointXYZRGBNormal, pcl::PointXYZ>(cloud, neighbors, octree, visited, next_elem);
            filter_PF_neighbors_point<std::vector<int>>(next_elem, cloud, ls, visited, visited[next_elem], sigma_s, sigma_r, sigma_d);
            counter++;
            //std::cout << counter << std::endl;
        }
    }

    for (int i:order_visit) {
        results[0]->points[i] = ls[0]->points[i];
        results[1]->points[i] = ls[2]->points[i];
    }
}

void BFS_back_point(pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr cloud, std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr>& results, std::vector<int>& order_visit,
        pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>& octree, float sigma_s, float sigma_r, float sigma_d) {
    std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> ls(4);
    for (int i=0; i < 3; i+=2) {
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr l (new pcl::PointCloud<pcl::PointXYZRGBNormal>());
        ls[i] = l;
        pcl::copyPointCloud(*cloud, *ls[i]);
        initialize<pcl::PointXYZRGBNormal>(ls[i], 0);
    }

    for (int i=1; i < 4; i+=2) {
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr l (new pcl::PointCloud<pcl::PointXYZRGBNormal>());
        ls[i] = l;
        pcl::copyPointCloud(*cloud, *ls[i]);
        initialize<pcl::PointXYZRGBNormal>(ls[i], 0.000001);
    }

    std::deque<int> neighbors;

    std::unordered_map<int, std::vector<int>> visited;
    for (int i=order_visit.size() - 1; i >= 0; i--) {
        int next_elem = order_visit[i];
        copy_n_point<pcl::PointXYZRGBNormal, pcl::PointXYZ>(cloud, neighbors, octree, visited, next_elem);
        filter_PF_neighbors_point<std::vector<int>>(next_elem, cloud, ls, visited, visited[next_elem], sigma_s, sigma_r, sigma_d);
    }
    for (int i=0; i < cloud->size(); i++) {
        results[0]->points[i] = ls[0]->points[i];
        results[1]->points[i] = ls[2]->points[i];
    }

}

void pf_3D_point(pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr cloud, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr result, int iter, bool back, float sigma_s, float sigma_r, float sigma_d) {
    float resolution = 128.0f;
    std::vector<int> order_visit;
    order_visit.reserve(cloud->size());

    pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree (resolution);

    pcl::PointCloud<pcl::PointXYZ>::Ptr octree_aux (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(*cloud, *octree_aux);

    octree.setInputCloud (octree_aux);
    octree.addPointsFromInputCloud ();

    pcl::Normal one(1.f,1.f,1.f);
    pcl::PointXYZRGBNormal num, den;

    std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> ls(2);
    for (int i=0; i < 2; i++) {
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr l (new pcl::PointCloud<pcl::PointXYZRGBNormal>());
        ls[i] = l;
        pcl::copyPointCloud(*cloud, *ls[i]);
        initialize<pcl::PointXYZRGBNormal>(ls[i], 0);
    }

    std::vector<pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr> rs(2);
    for (int i=0; i < 2; i++) {
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr l (new pcl::PointCloud<pcl::PointXYZRGBNormal>());
        rs[i] = l;
        pcl::copyPointCloud(*cloud, *rs[i]);
        initialize<pcl::PointXYZRGBNormal>(rs[i], 0);
    }

    for (int i=0; i < iter; i++) {

        std::cout << "Beginning" << std::endl;
        BFS_point(result, ls, order_visit, octree, sigma_s, sigma_r, sigma_d);
        std::cout << "Half" << std::endl;
        if (back) {
            BFS_back_point(result, rs, order_visit, octree, sigma_s, sigma_r, sigma_d);
        }
        

        for (int i=0; i < cloud->size(); i++) {
            //std::cout << "P: " << i << " " << cloud->points[i] << std::endl;
            num.getNormalVector3fMap() = ls[0]->points[i].getNormalVector3fMap() + rs[0]->points[i].getNormalVector3fMap();
            den.getNormalVector3fMap() =  ls[1]->points[i].getNormalVector3fMap() + one.getNormalVector3fMap() + rs[1]->points[i].getNormalVector3fMap();
            
            //std::cout << "num: " << num << " den: " << den <<std::endl;

            pcl::PointXYZRGBNormal aux;
            divide_normal<pcl::PointXYZRGBNormal>(num, den, aux);

            aux.normal_x *= cloud->points[i].normal_x;
            aux.normal_y *= cloud->points[i].normal_y;
            aux.normal_z *= cloud->points[i].normal_z;
            //aux.getNormalVector3fMap() = cloud->points[i].getNormalVector3fMap()*aux.getNormalVector3fMap();

            sum_normal_point<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>(aux, result->points[i], result->points[i]);
            //result->points[i].getArray3fMap() += aux.getNormalVector3fMap()*cloud->points[i].getNormalVector3fMap();

            //std::cout << "R: " << i << " " << result->points[i] << std::endl;
        }
    }
}