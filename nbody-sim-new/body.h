#ifndef BODY_H
#define BODY_H

#include "vector.h"

// Generic Body class that works with any dimension
template <int D>
struct Body {
    Vector<D> position;
    Vector<D> velocity;
    double mass;

    Body() : position(), velocity(), mass(0.0) {}
    
    Body(const Vector<D>& pos, double m) : position(pos), velocity(), mass(m) {}
    
    Body(const Vector<D>& pos, const Vector<D>& vel, double m) 
        : position(pos), velocity(vel), mass(m) {}
};

// Type aliases for convenience
using Body2D = Body<2>;
using Body3D = Body<3>;

#endif // BODY_H
