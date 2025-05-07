// FMM_OMP.tpp

template <int D>
void FMM_OMP<D>::p2m_phase() {
    if (nodes_by_level.empty()) return;
    
    // Get leaf nodes (last level might not all be leaves in adaptive tree)
    std::vector<FMMNode<D>*> leaf_nodes;
    
    for (size_t level = 0; level < nodes_by_level.size(); ++level) {
        for (FMMNode<D>* node : nodes_by_level[level]) {
            if (node && node->is_leaf()) {
                leaf_nodes.push_back(node);
            }
        }
    }
    
    // Parallelize P2M computation for all leaf nodes
    #pragma omp parallel for
    for (size_t i = 0; i < leaf_nodes.size(); ++i) {
        FMMNode<D>* node = leaf_nodes[i];
        
        // Clear multipole
        node->multipole.clear();
        
        if constexpr (D == 2) {
            // 2D implementation with complex numbers
            double total_mass = 0.0;
            Vector<D> center_of_mass = Vector<D>();
            
            for (Body<D>* body : node->bodies) {
                total_mass += body->mass;
                center_of_mass += body->position * body->mass;
            }
            
            if (total_mass > 0) {
                center_of_mass = center_of_mass / total_mass;
            }
            
            node->multipole.coeff[0] = std::complex<double>(total_mass, 0.0);
            
            if (order > 0) {
                std::complex<double> z_com = to_complex(center_of_mass - node->center);
                node->multipole.coeff[1] = z_com * total_mass;
            }
            
            for (Body<D>* body : node->bodies) {
                std::complex<double> z = to_complex(body->position - node->center);
                std::complex<double> z_power = z * z;
                
                for (int p = 2; p <= order; ++p) {
                    node->multipole.coeff[p] += -body->mass * z_power / static_cast<double>(p);
                    z_power *= z;
                }
            }
        } else {
            double total_mass = 0.0;
            Vector<D> center_of_mass = Vector<D>();
            
            for (Body<D>* body : node->bodies) {
                total_mass += body->mass;
                center_of_mass += body->position * body->mass;
            }
            
            if (total_mass > 0) {
                center_of_mass = center_of_mass / total_mass;
            }
            
            node->multipole.coeff[0] = std::complex<double>(total_mass, 0.0);
            
            if (order > 0 && node->multipole.coeff.size() > 2) {
                node->multipole.coeff[1] = std::complex<double>(center_of_mass[0], center_of_mass[1]);
                node->multipole.coeff[2] = std::complex<double>(center_of_mass[2], 0.0);
            }
        }
    }
}

template <int D>
void FMM_OMP<D>::m2l_phase() {
    if (nodes_by_level.empty()) return;
    
    std::vector<FMMNode<D>*> nodes_with_interactions;
    
    for (auto& level_nodes : nodes_by_level) {
        for (FMMNode<D>* node : level_nodes) {
            if (node && !node->interaction_list.empty()) {
                nodes_with_interactions.push_back(node);
            }
        }
    }
    
    #pragma omp parallel for
    for (size_t i = 0; i < nodes_with_interactions.size(); ++i) {
        FMMNode<D>* node = nodes_with_interactions[i];
        
        node->local.clear();
        
        for (FMMNode<D>* source : node->interaction_list) {
            if (!source) continue;
            
            double src_mass = source->multipole.coeff[0].real();
            if (src_mass < 1e-10) continue;
            
            if constexpr (D == 2) {
                std::complex<double> z0 = to_complex(source->center - node->center);
                double r = std::abs(z0);
                
                if (r < 1e-10) continue;
                
                std::complex<double> source_com = z0;
                if (order > 0 && std::abs(source->multipole.coeff[1]) > 1e-10) {
                    source_com = source->multipole.coeff[1] / src_mass + z0;
                }
                
                for (int p = 0; p <= order; ++p) {
                    for (int q = 0; q <= order; ++q) {
                        std::complex<double> term;
                        if (q == 0) {
                            if (p == 0) {
                                term = src_mass / r;
                            } else {
                                term = src_mass * std::pow(1.0 / source_com, p);
                            }
                        } else if (p + q <= order) {
                            term = source->multipole.coeff[q] * 
                                  std::pow(1.0 / z0, p + q) *
                                  static_cast<double>(binomial(p + q - 1, q - 1));
                        }
                        node->local.coeff[p] += term;
                    }
                }
            } else {
                Vector<D> r_vec = source->center - node->center;
                double r = r_vec.magnitude();
                
                if (r < 1e-10) continue;
                
                Vector<D> source_com = source->center;
                if (source->multipole.coeff.size() > 2) {
                    source_com[0] = source->multipole.coeff[1].real();
                    source_com[1] = source->multipole.coeff[1].imag();
                    source_com[2] = source->multipole.coeff[2].real();
                }
                
                Vector<D> r_com_vec = source_com - node->center;
                double r_com = r_com_vec.magnitude();
                
                if (r_com > 1e-10) {
                    node->local.coeff[0] += std::complex<double>(src_mass / r_com, 0.0);
                    
                    if (node->local.coeff.size() > 2) {
                        node->local.coeff[1] = std::complex<double>(r_com_vec[0], r_com_vec[1]);
                        node->local.coeff[2] = std::complex<double>(r_com_vec[2], r_com);
                    }
                }
            }
        }
    }
}

template <int D>
void FMM_OMP<D>::l2p_phase(std::vector<Vector<D>>& forces, const std::vector<Body<D>>& bodies) {
    if (nodes_by_level.empty()) return;
    
    std::vector<FMMNode<D>*> leaf_nodes;
    
    for (auto& level_nodes : nodes_by_level) {
        for (FMMNode<D>* node : level_nodes) {
            if (node && node->is_leaf()) {
                leaf_nodes.push_back(node);
            }
        }
    }
    
    #pragma omp parallel for
    for (size_t i = 0; i < leaf_nodes.size(); ++i) {
        FMMNode<D>* node = leaf_nodes[i];
        if (!node || node->bodies.empty()) continue;
        
        for (Body<D>* body_ptr : node->bodies) {
            size_t body_idx = 0;
            for (size_t j = 0; j < bodies.size(); ++j) {
                if (&bodies[j] == body_ptr) {
                    body_idx = j;
                    break;
                }
            }
            
            if constexpr (D == 2) {
                std::complex<double> z = to_complex(body_ptr->position - node->center);
                std::complex<double> potential(0.0, 0.0);
                std::complex<double> potential_gradient(0.0, 0.0);
                
                for (int p = 1; p <= order; ++p) {
                    if (std::abs(node->local.coeff[p]) < 1e-15) continue;
                    
                    std::complex<double> z_power;
                    if (p == 1) {
                        z_power = std::complex<double>(1.0, 0.0);
                    } else {
                        z_power = pow(z, p-1);
                    }
                    
                    potential_gradient += node->local.coeff[p] * static_cast<double>(p) * z_power;
                    potential += node->local.coeff[p] * pow(z, p);
                }
                
                Vector<D> force;
                force[0] = -potential_gradient.real() * body_ptr->mass;
                force[1] = -potential_gradient.imag() * body_ptr->mass;
                
                forces[body_idx] += force;
            } else {
                double potential = node->local.coeff[0].real();
                Vector<D> r_vec;
                double r = 1.0;
                
                if (node->local.coeff.size() > 2) {
                    r_vec[0] = node->local.coeff[1].real();
                    r_vec[1] = node->local.coeff[1].imag();
                    r_vec[2] = node->local.coeff[2].real();
                    r = node->local.coeff[2].imag();
                    
                    if (r > 1e-10) {
                        r_vec = r_vec / r;
                        
                        double force_mag = G * body_ptr->mass * potential / r;
                        
                        forces[body_idx] -= r_vec * force_mag;
                    }
                }
            }
        }
    }
}

template <int D>
std::vector<Vector<D>> FMM_OMP<D>::calculate_forces(const std::vector<Body<D>>& bodies) {
    std::vector<Vector<D>> forces(bodies.size(), Vector<D>());
    
    if (!root) return forces;
    
    p2m_phase();
    m2m_phase();
    m2l_phase();
    l2l_phase();
    l2p_phase(forces, bodies);
    p2p_phase(forces, bodies);
    
    return forces;
}