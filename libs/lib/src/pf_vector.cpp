#include "pf_vector.hpp"

#include <pcl/octree/octree_search.h>
#include <pcl/common/common_headers.h>
#include <pcl/filters/bilateral.h>
#include <pcl/filters/fast_bilateral.h>
#include <pcl/common/geometry.h>
#include <pcl/surface/gp3.h>
#include <chrono>

#include <Eigen/Core>
#include <cmath>
#include <unordered_map>

#include "pf.hpp"
#include "util.hpp"

//Adapt to consider vectors
void SpatialFilter::copy_n_vec(std::deque<int>& neighbors_next, int current_elem) {
    //for (int point: this->neighbors[current_elem])
    //    this->neighbors.push_back(point);
    //std::cout << "N# " << this->neighbors[current_elem].size() << std::endl;
    neighbors_next.insert(neighbors_next.end(), this->neighbors[current_elem].begin(), this->neighbors[current_elem].end());
    this->visited[current_elem] = 1;
}

inline
float SpatialFilter::permeability_radial_vec(const pcl::PointXYZRGBNormal& p, const  pcl::PointXYZRGBNormal& pn,
 const pcl::PointXYZRGBNormal& n, const pcl::PointXYZRGBNormal& nn) {
    float cos_val = dot_product_normal<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>(pn, nn);// pn.getNormalVector3fMap().dot(nn.getNormalVector3fMap());
    //float cos_val = nn_normal.dot (pn_normal);
    float fdist = acos(std::max(std::min(1.0f, cos_val), -1.0f));

    float denominator = M_PI*sigma_r;
    float aux = pow(fdist/denominator, sigma_s);

    pcl::PointXYZ diff(p.x - n.x, p.y - n.y, p.z - n.z);
    
    float ddist = norm<pcl::PointXYZ>(diff);
    return std::min(1/(1 + aux), 1.0f);
}

void SpatialFilter::filter_PF_vec(int index, int neighbor_idx,
    const pcl::PointXYZINormal *current,
    const pcl::PointXYZRGBNormal *current_point,
    pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr cloud,
    pcl::PointCloud<pcl::PointXYZINormal>::ConstPtr vecs,
    std::vector<std::vector<pcl::PointXYZINormal>> &s) {

    const pcl::PointXYZRGBNormal &neighbor_point = cloud->points[neighbor_idx];
    const pcl::PointXYZRGBNormal &neighbor_normal = cloud->points[neighbor_idx];

    float permeability = permeability_radial_vec(*current_point, *current_point, neighbor_point, neighbor_normal);
    //std::cout << "Perm " << permeability << std::endl;
    float sq_per = permeability*permeability;

    pcl::PointXYZINormal previous = s[0][neighbor_idx];
    pcl::PointXYZINormal& s_0_n = s[0][neighbor_idx], s_0_i = s[0][index];

    
    sum_multi_normal2<pcl::PointXYZINormal>(*current, s_0_i, sq_per, s_0_n);
    
    s_0_n.intensity += sq_per*(current->intensity + s_0_i.intensity);
    

    sum_multi_scalar4d<pcl::PointXYZINormal>(s[1][neighbor_idx], 1, permeability, s[1][neighbor_idx]);
    
    pcl::PointXYZINormal& s_2_n = s[2][neighbor_idx], s_2_i = s[2][index];

    sum_multi_normal2<pcl::PointXYZINormal>(s_2_i, this->one, sq_per, s_2_n);
    s_2_n.intensity += sq_per*(s_2_i.intensity + 1.0f);

    sum_multi_scalar4d<pcl::PointXYZINormal>(s[3][neighbor_idx], 1, permeability, s[3][neighbor_idx]);
    
}

void SpatialFilter::filter_PF_neighbors_vec(int index,
    pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr cloud,
    pcl::PointCloud<pcl::PointXYZINormal>::ConstPtr vecs,
    std::vector<std::vector<pcl::PointXYZINormal>> &s) {
    pcl::PointXYZINormal previous = s[0][index], previous2 = s[2][index];
    divide_normal4d<pcl::PointXYZINormal>(s[0][index], s[1][index], s[0][index]);
    
    divide_normal4d<pcl::PointXYZINormal>(s[2][index], s[3][index], s[2][index]);
    
    const pcl::PointXYZINormal *current = &vecs->points[index];
    const pcl::PointXYZRGBNormal *current_point = &cloud->points[index];

    for (size_t j = 0; j < this->neighbors[index].size(); ++j) {
        int neighbor_idx = this->neighbors[index][j];
        if ((neighbor_idx != index)  && this->visited[neighbor_idx] == -1) {
            this->filter_PF_vec(index, neighbor_idx, current, current_point, cloud, vecs, s);
        }
    }    
}


void SpatialFilter::BFS_back_vec(pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr cloud,
    pcl::PointCloud<pcl::PointXYZINormal>::ConstPtr vecs) {   
 
    std::deque<int> next_neighbors;

    //std::vector<int> initial_points = {3251, 3110, 3103, 236, 3108, 2876, 2877, 1427, 3252};
    std::cout << "starting back "  << order_visit.size() << std::endl;
    for (int i=order_visit.size() - 1; i >= 0; i--) {
        int next_elem = this->order_visit[i];
    //for (int next_elem:initial_points) {
        this->copy_n_vec(next_neighbors, next_elem);
        this->filter_PF_neighbors_vec(next_elem, cloud, vecs, this->rs);

        //std::cout << "Elem: " << next_elem << " " << this->rs[0][next_elem] << std::endl;
    }
}

void SpatialFilter::initialize_s(pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr cloud) {
    int cloud_size = cloud->points.size();
    pcl::PointXYZINormal empty;
    for (int i=0; i < 3; i+=2) {
        this->ls[i] = std::vector<pcl::PointXYZINormal>(cloud_size, empty);
        this->rs[i] = std::vector<pcl::PointXYZINormal>(cloud_size, empty);
    }
    initialize4d<pcl::PointXYZINormal>(empty, 0.000001);
    for (int i=1; i < 4; i+=2) {
        this->ls[i] = std::vector<pcl::PointXYZINormal>(cloud_size, empty);
        this->rs[i] = std::vector<pcl::PointXYZINormal>(cloud_size, empty);
    }
}

void SpatialFilter::reset_s() {
    pcl::PointXYZINormal empty;
    for (int i=0; i < 3; i+=2) {
        std::fill(this->ls[i].begin(), this->ls[i].end(), empty);
        std::fill(this->rs[i].begin(), this->rs[i].end(), empty);
    }
    initialize4d<pcl::PointXYZINormal>(empty, 0.000001);
    for (int i=1; i < 4; i+=2) {
        std::fill(this->ls[i].begin(), this->ls[i].end(), empty);
        std::fill(this->rs[i].begin(), this->rs[i].end(), empty);
    }
}

void SpatialFilter::BFS_vec(pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr cloud,
    pcl::PointCloud<pcl::PointXYZINormal>::ConstPtr vecs,
    int initial_point) {

    int cloud_size = cloud->points.size();

    std::deque<int> next_neighbors;
    int counter = 0;
    this->order_visit.push_back(initial_point);
    this->copy_n_vec(next_neighbors, initial_point);
    this->filter_PF_neighbors_vec(initial_point, cloud, vecs, this->ls);
    for (int i = initial_point;;) {
        while (next_neighbors.size() != 0 && counter < 300000) {
            int next_elem = next_neighbors.front();
            next_neighbors.pop_front();
            if (this->visited[next_elem] == -1) {
                counter++;
                this->order_visit.push_back(next_elem);
                this->copy_n_vec(next_neighbors, next_elem);
                this->filter_PF_neighbors_vec(next_elem, cloud, vecs, this->ls);
            }
            //return;
        }
        if (i == cloud->size()-1)
            i = 0;
        else
            i++;

        if (i == initial_point)
            break;
        
        if (this->visited[i] == -1) {
            next_neighbors.push_back(i);
        }
    }
}


void SpatialFilter::get_neighbors
(pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr cloud) {

    pcl::search::KdTree<pcl::PointXYZRGBNormal> tree;
    tree.setInputCloud (cloud);

    int point_number = static_cast<int> (cloud->size ());

    std::vector<int> pre_alloc_point_neighbors;
    this->neighbors.resize (point_number, pre_alloc_point_neighbors);
    this->order_visit.reserve(point_number);
    //std::cout << "radius " << radius << std::endl;
    float sum_n = 0;
    #pragma omp parallel for
    for (int i = 0; i < point_number; i++)
    {
        std::vector<int> point_neighbors;
        std::vector<float> distances;
        //point_neighbors.clear();
        tree.radiusSearch (i, radius, point_neighbors, distances);
        // Could limit number points inside range. This would make things faster
        this->neighbors[i].swap (point_neighbors);
        sum_n += distances.size();
    }
    std::cout << "Avg N " << sum_n/float(point_number) << std::endl;
}

void SpatialFilter::reset() {
    this->neighbors.clear();
    this->order_visit.clear();
}

void SpatialFilter::pf_3D_vec(pcl::PointCloud<pcl::PointXYZRGBNormal>::ConstPtr cloud
    , pcl::PointCloud<pcl::PointXYZINormal>::Ptr vecs
    , pcl::PointCloud<pcl::PointXYZINormal>::Ptr result
    , const std::vector<int>& initial_points) {

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    this->get_neighbors(cloud);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    double elapsed_secs = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    std::cout << "Elapsed N time: " << elapsed_secs << std::endl;

    pcl::copyPointCloud(*vecs, *result);

    std::cout << "Filter start " << cloud->size() << std::endl;
    this->initialize_s(cloud);

    int cloud_size = cloud->points.size();
    this->visited = std::vector<int>(cloud_size, -1);

    pcl::PointXYZINormal one;
    one.normal_x = one.normal_y = one.normal_z = 1.f;
    pcl::PointXYZINormal num, den;

    std::chrono::steady_clock::time_point begin2 = std::chrono::steady_clock::now();
    for(int j=0; j < std::min(this->number_initial, (int)initial_points.size()); j++) {
        int initial = initial_points[j];

        begin = std::chrono::steady_clock::now();
        std::cout << "BFS " << initial << std::endl;
        BFS_vec(cloud, result, initial);
        std::fill(this->visited.begin(), this->visited.end(), -1);
        end = std::chrono::steady_clock::now();
        elapsed_secs = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
        std::cout << "Elapsed F time: " << elapsed_secs << std::endl;

        begin = std::chrono::steady_clock::now();
        std::cout << "Half" << std::endl;
        BFS_back_vec(cloud, result);
        std::fill(this->visited.begin(), this->visited.end(), -1);
        end = std::chrono::steady_clock::now();
        elapsed_secs = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
        std::cout << "Elapsed B time: " << elapsed_secs << std::endl;

        begin = std::chrono::steady_clock::now();
        std::cout << "Going to norm" << std::endl;
        for (int i=0; i < cloud_size; i++) {
        //for (int i:initial_points) {

            sum3<pcl::PointXYZINormal>(this->ls[0][i], vecs->points[i], this->rs[0][i], num);
            //num.getNormalVector3fMap() = this->ls[0][i].getNormalVector3fMap() + vecs->points[i].getNormalVector3fMap() + this->rs[0][i].getNormalVector3fMap();
            num.intensity = this->ls[0][i].intensity + vecs->points[i].intensity + this->rs[0][i].intensity;
            
            sum3<pcl::PointXYZINormal>(this->ls[2][i], one, this->rs[2][i], den);
            //den.getNormalVector3fMap() = this->ls[2][i].getNormalVector3fMap() + one.getNormalVector3fMap() + this->rs[2][i].getNormalVector3fMap();
            den.intensity = this->ls[2][i].intensity + 1.0f + this->rs[2][i].intensity;

            divide_normal<pcl::PointXYZINormal>(num, den, result->points[i]);
            result->points[i].intensity = num.intensity/den.intensity;
        }
        std::cout << "Going to reset " << j << " " <<  this->number_initial << std::endl;
        this->reset_s();
        this->order_visit.clear();
        end = std::chrono::steady_clock::now();
        elapsed_secs = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
        std::cout << "Elapsed No time: " << elapsed_secs << std::endl;
    }
    std::chrono::steady_clock::time_point end2 = std::chrono::steady_clock::now();
    elapsed_secs = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - begin2).count();
    std::cout << "Elapsed F time: " << elapsed_secs << std::endl;
}
