#ifndef UTILS_H
#define UTILS_H

#include <random>
#include <vector>
#define grav 6.6743e-11

struct Body {
    double x;
    double y;
    double mass;
    Body(double x = 0.0, double y = 0.0, double mass = 0.0);
};

struct Vector2D {
    double x;
    double y;
    Vector2D(double x = 0.0, double y = 0.0);
};

std::vector<Body> generate_random_bodies(int N);

#endif