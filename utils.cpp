#include "utils.h"
using namespace std;

Body::Body(double x, double y, double mass) {
    this->x = x;
    this->y = y;
    this->mass = mass;
}

Vector2D::Vector2D(double x, double y) {
    this->x = x;
    this->y = y;
}

vector<Body> generate_random_bodies(int N) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> mass_dis(10.0, 10000.0);
    std::uniform_int_distribution<> pos_dis(0.0, 1000000.0);
    vector<Body> bodies(N);
    for (int i = 0; i < N; i++) {
        double random_x = pos_dis(gen);
        double random_y = pos_dis(gen);
        double random_mass = mass_dis(gen);
        bodies[i].x = random_x;
        bodies[i].y = random_y;
        bodies[i].mass = random_mass;
    }
    return bodies;
}