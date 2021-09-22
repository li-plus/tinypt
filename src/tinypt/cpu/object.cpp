#include "tinypt/cpu/object.h"

#include <filesystem>
#include <tinyobjloader/tiny_obj_loader.h>

namespace tinypt {
namespace cpu {

namespace fs = std::filesystem;

static inline float plane_pdf_value(float dist, float area, const Vec3f &normal, const Vec3f &ray_dir) {
    float dist_squared = square(dist);
    float cos_alpha = std::abs(normal.dot(ray_dir));
    float pdf_val = dist_squared / (cos_alpha * area); // inf!
    return (pdf_val < INF) ? pdf_val : INF;
}

void Sphere::intersect(const Ray &ray, Hit &hit, RandEngine &rng) const { intersect_impl(ray, hit, &rng); }
void Sphere::intersect_t(const Ray &ray, Hit &hit) const { intersect_impl<false>(ray, hit, nullptr); }

template <bool ENABLE_SURFACE>
void Sphere::intersect_impl(const Ray &ray, Hit &hit, RandEngine *rng) const {
    DCHECK(is_normalized(ray.dir));
    Vec3f op = _center - ray.org;
    float b = op.dot(ray.dir);
    float det = square(b) - op.dot(op) + square(_radius);
    if (det < 0) {
        return;
    }
    det = std::sqrt(det);
    float t;
    bool into;
    if (b - det >= EPS) {
        t = b - det;
        into = true;
    } else if (b + det >= EPS) {
        t = b + det;
        into = false;
    } else {
        t = INF;
        into = true;
    }
    if (t < hit.t) {
        if constexpr (ENABLE_SURFACE) {
            Vec3f pos = ray.point_at(t);
            Vec3f normal = (pos - _center).normalized();
            // TODO: merge codes
            float u = (std::atan2(normal.y(), normal.x()) + M_PIf32) * (.5f * M_1_PIf32);
            float v = std::acos(normal.z()) * M_1_PIf32;
            DCHECK(0 <= u && u <= 1) << u;
            DCHECK(0 <= v && v <= 1) << v;
            Vec2f uv(u, v);
            if (_material.alpha_texture().alpha_at(uv) < rng->random()) {
                return;
            }
            if (!into) {
                normal = -normal;
            }
            hit = Hit(t, normal, normal, into, uv, &_material);
        } else {
            hit.t = t;
        }
    }
}

Vec3f Sphere::sample(const Vec3f &org, RandEngine &rng) const {
    float cos_theta_max = get_cos_theta_max(org);
    float cos_theta = rng.random(cos_theta_max, 1);
    float sin_theta = std::sqrt(1 - cos_theta * cos_theta);
    DCHECK(std::isfinite(sin_theta));
    float phi = rng.random(2.f * M_PIf32);
    Vec3f local_dir(sin_theta * std::cos(phi), sin_theta * std::sin(phi), cos_theta);
    Vec3f out_dir = Rotation::from_direction((_center - org).normalized()) * local_dir;
    return out_dir;
}

float Sphere::pdf(const Ray &ray) const {
    Hit hit;
    intersect_impl<false>(ray, hit, nullptr);
    if (!hit.is_hit()) {
        return 0;
    }
    float cos_theta_max = get_cos_theta_max(ray.org);
    float solid_angle = 2 * M_PIf32 * (1 - cos_theta_max);
    float pdf_val = 1 / solid_angle;
    DCHECK(pdf_val < INF);
    return pdf_val;
}

AABB Sphere::bounding_box() const {
    Vec3f offset(_radius, _radius, _radius);
    return {_center - offset, _center + offset};
}

float Sphere::get_cos_theta_max(const Vec3f &org) const {
    float dist2 = (_center - org).squaredNorm();
    float radius2 = square(_radius);
    if (dist2 <= radius2) {
        // ray origin inside the sphere, sample the entire surface
        return -1;
    }
    float cos_theta_max = std::sqrt(1 - radius2 / dist2);
    return cos_theta_max;
}

Rectangle::Rectangle(const Vec3f &location, const Mat3f &rotation, const Vec2f &dimension, Material material)
    : _center(location), _rotation(rotation), _dimension(dimension), _material(std::move(material)),
      _dist(calc_dist(_rotation, _center)) {}

void Rectangle::intersect(const Ray &ray, Hit &hit, RandEngine &rng) const { intersect_impl(ray, hit, &rng); }

void Rectangle::intersect_t(const Ray &ray, Hit &hit) const { intersect_impl(ray, hit, nullptr); }

template <bool ENABLE_SURFACE>
void Rectangle::intersect_impl(const Ray &ray, Hit &hit, RandEngine *rng) const {
    DCHECK(is_normalized(ray.dir));
    float t = (_dist - ray.org.dot(normal())) / ray.dir.dot(normal()); // !inf
    if (EPS <= t && t < hit.t) {
        Vec3f cp = ray.point_at(t) - _center; // center -> hit_pos
        float u = cp.dot(_rotation.u()) / _dimension.x() + 0.5f;
        float v = cp.dot(_rotation.v()) / _dimension.y() + 0.5f;
        if (0 <= u && u <= 1 && 0 <= v && v <= 1) {
            if constexpr (ENABLE_SURFACE) {
                Vec3f hit_normal = is_same_side(normal(), ray.dir) ? -normal() : normal();
                hit = Hit(t, hit_normal, hit_normal, true, {u, v}, &_material);
            } else {
                hit.t = t;
            }
        }
    }
}

Vec3f Rectangle::sample(const Vec3f &org, RandEngine &rng) const {
    Vec3f u_vec = (rng.random() - .5f) * _dimension.x() * _rotation.u();
    Vec3f v_vec = (rng.random() - .5f) * _dimension.y() * _rotation.v();
    Vec3f target = _center + u_vec + v_vec;
    return (target - org).normalized();
}

float Rectangle::pdf(const Ray &ray) const {
    Hit hit;
    intersect_t(ray, hit);
    if (!hit.is_hit()) {
        return 0;
    }
    float area = _dimension.x() * _dimension.y();
    return plane_pdf_value(hit.t, area, normal(), ray.dir);
}

AABB Rectangle::bounding_box() const {
    AABB bbox({-_dimension.x() * .5f, -_dimension.y() * .5f, -EPS}, {_dimension.x() * .5f, _dimension.y() * .5f, EPS});
    bbox = Mapping(Vec3f::Zero(), _rotation.matrix()).map_bounding_box(bbox);
    return bbox;
}

Circle::Circle(float radius, const Vec3f &location, const Mat3f &rotation, Material material)
    : _radius(radius), _center(location), _rot(rotation), _material(std::move(material)), _dist(_rot.w().dot(_center)) {
}

void Circle::intersect(const Ray &ray, Hit &hit, RandEngine &rng) const { intersect_impl(ray, hit, &rng); }
void Circle::intersect_t(const Ray &ray, Hit &hit) const { intersect_impl<false>(ray, hit, nullptr); }

template <bool ENABLE_SURFACE>
void Circle::intersect_impl(const Ray &ray, Hit &hit, RandEngine *rng) const {
    DCHECK(is_normalized(ray.dir));
    float t = (_dist - ray.org.dot(normal())) / ray.dir.dot(normal()); // !inf
    if (EPS <= t && t < hit.t) {
        Vec3f cp = ray.point_at(t) - _center; // center -> hit_pos
        if (cp.squaredNorm() <= square(_radius)) {
            if constexpr (ENABLE_SURFACE) {
                Vec3f hit_normal = is_same_side(normal(), ray.dir) ? -normal() : normal();
                cp /= _radius;
                Vec2f uv(cp.dot(_rot.u()), cp.dot(_rot.v()));
                hit = Hit(t, hit_normal, hit_normal, true, uv, &_material);
            } else {
                hit.t = t;
            }
        }
    }
}

AABB Circle::bounding_box() const {
    AABB bbox({-_radius, -_radius, -EPS}, {_radius, _radius, EPS});
    bbox = Mapping(location(), rotation()).map_bounding_box(bbox);
    return bbox;
}

Vec3f Circle::sample(const Vec3f &org, RandEngine &rng) const {
    Vec2f uv = rng.rand_in_disk(_radius);
    Vec3f target = _center + _rot.u() * uv.x() + _rot.v() * uv.y();
    DCHECK(is_close(target.dot(normal()), _dist));
    return (target - org).normalized();
}

float Circle::pdf(const Ray &ray) const {
    Hit hit;
    intersect_impl<false>(ray, hit, nullptr);
    if (!hit.is_hit()) {
        return 0;
    }
    float area = M_PIf32 * square(_radius);
    return plane_pdf_value(hit.t, area, normal(), ray.dir);
}

TriangleMeshFace::TriangleMeshFace(const Vec3f &a, const Vec3f &b, const Vec3f &c, int face_id,
                                   const TriangleMesh *mesh)
    : _a(a), _ab(b - a), _ac(c - a), _face_id(face_id), _mesh(mesh) {}

void TriangleMeshFace::intersect(const Ray &ray, Hit &hit, RandEngine &rng) const { intersect_impl(ray, hit, &rng); }
void TriangleMeshFace::intersect_t(const Ray &ray, Hit &hit) const { intersect_impl<false>(ray, hit, nullptr); }

const Material &TriangleMeshFace::material() const {
    int material_id = _mesh->_material_ids[_face_id];
    return _mesh->_materials[material_id];
}

AABB TriangleMeshFace::bounding_box() const {
    Vec3f b = vb();
    Vec3f c = vc();
    Vec3f minv = _a.cwiseMin(b).cwiseMin(c);
    Vec3f maxv = _a.cwiseMax(b).cwiseMax(c);
    AABB bbox(minv, maxv);
    return bbox;
}

Vec3f TriangleMeshFace::sample(const Vec3f &org, RandEngine &rng) const {
    // https://math.stackexchange.com/questions/18686/uniform-random-point-in-triangle-in-3d
    // TODO: why?
    float sqrt_r1 = std::sqrt(rng.random());
    float r2 = rng.random();
    float u = sqrt_r1 * (1 - r2);
    float v = r2 * sqrt_r1;
    Vec3f target = _a + u * _ab + v * _ac;
    return (target - org).normalized();
}

float TriangleMeshFace::pdf(const Ray &ray) const {
    DCHECK(is_normalized(ray.dir));
    Hit hit;
    intersect_impl<false>(ray, hit, nullptr);
    if (!hit.is_hit()) {
        return 0;
    }
    Vec3f face_normal = _ac.cross(_ab);
    float area = .5f * face_normal.norm();
    face_normal.normalize();
    return plane_pdf_value(hit.t, area, face_normal, ray.dir);
}

template <bool ENABLE_SURFACE>
void TriangleMeshFace::intersect_impl(const Ray &ray, Hit &hit, RandEngine *rng) const {
    float t, u, v;
    {
        Vec3f p = ray.dir.cross(_ac);
        float inv_det = 1.f / p.dot(_ab); // inf!
        Vec3f ao = ray.org - _a;
        u = p.dot(ao) * inv_det;
        Vec3f q = ao.cross(_ab);
        v = q.dot(ray.dir) * inv_det;
        if (!(0 <= u && 0 <= v && u + v <= 1)) {
            return;
        }
        t = q.dot(_ac) * inv_det;
    }

    if (EPS <= t && t < hit.t) {
        if constexpr (ENABLE_SURFACE) {
            const auto &face_index = _mesh->_faces_index[_face_id];
            const auto &idx_a = face_index[0];
            const auto &idx_b = face_index[1];
            const auto &idx_c = face_index[2];

            Vec2f uv;
            if (idx_a.texcoord_index >= 0) {
                // vertex texture coordinates
                const Vec2f &uv_a = _mesh->_texture_coords[idx_a.texcoord_index];
                const Vec2f &uv_b = _mesh->_texture_coords[idx_b.texcoord_index];
                const Vec2f &uv_c = _mesh->_texture_coords[idx_c.texcoord_index];
                uv = (1 - u - v) * uv_a + u * uv_b + v * uv_c;
            } else {
                uv = {u, v};
            }
            auto mat = &material();
            // transparent mask
            if (mat->alpha_texture().alpha_at(uv) < rng->random()) {
                return;
            }

            Vec3f normal;
            if (idx_a.normal_index >= 0) {
                // vertex normal
                const Vec3f &vn_a = _mesh->_vertex_normals[idx_a.normal_index];
                const Vec3f &vn_b = _mesh->_vertex_normals[idx_b.normal_index];
                const Vec3f &vn_c = _mesh->_vertex_normals[idx_c.normal_index];
                normal = ((1 - u - v) * vn_a + u * vn_b + v * vn_c).normalized();
            } else {
                // face normal
                normal = _ac.cross(_ab).normalized();
            }

            bool into = !is_same_side(normal, ray.dir);
            if (!into) {
                normal = -normal;
            }
            Vec3f shade_normal = normal;

            const BumpTexture &bump_tex = mat->bump_texture();
            if (!bump_tex.empty()) {
                // compute dpdu, dpdv
                Vec2f duv_ab, duv_ac;
                if (idx_a.texcoord_index >= 0) {
                    // vertex texture coordinates
                    const Vec2f &uv_a = _mesh->_texture_coords[idx_a.texcoord_index];
                    const Vec2f &uv_b = _mesh->_texture_coords[idx_b.texcoord_index];
                    const Vec2f &uv_c = _mesh->_texture_coords[idx_c.texcoord_index];
                    duv_ab = uv_b - uv_a;
                    duv_ac = uv_c - uv_a;
                } else {
                    duv_ab = {1, 0};
                    duv_ac = {0, 1};
                }
                float du_ab = duv_ab.x(), dv_ab = duv_ab.y();
                float du_ac = duv_ac.x(), dv_ac = duv_ac.y();
                float inv_det = 1.f / (du_ab * dv_ac - du_ac * dv_ab); // !inf
                if (std::isfinite(inv_det)) {
                    Vec3f dpdu = (dv_ac * _ab - dv_ab * _ac) * inv_det;
                    Vec3f dpdv = (-du_ac * _ab + du_ab * _ac) * inv_det;
                    shade_normal = bump_tex.bump_normal(uv, shade_normal, dpdu, dpdv);
                }
            }
            hit = Hit(t, normal, shade_normal, into, uv, mat);
        } else {
            hit.t = t;
        }
    }
}

ObjectGroup::ObjectGroup(std::vector<std::shared_ptr<Object>> objects, int bvh_node_size)
    : _objects(std::move(objects)) {
    if (bvh_node_size > 0) {
        std::vector<Object *> obj_ptrs;
        for (const auto &obj : _objects) {
            obj_ptrs.emplace_back(obj.get());
        }
        _bvh = BVHTree(std::move(obj_ptrs), bvh_node_size);
    }
}

void ObjectGroup::intersect(const Ray &ray, Hit &hit, RandEngine &rng) const { intersect_impl(ray, hit, &rng); }
void ObjectGroup::intersect_t(const Ray &ray, Hit &hit) const { intersect_impl<false>(ray, hit, nullptr); }

Vec3f ObjectGroup::sample(const Vec3f &org, RandEngine &rng) const {
    DCHECK(!_objects.empty());
    const auto &obj = _objects[rng.rand_int(_objects.size())];
    return obj->sample(org, rng);
}

float ObjectGroup::pdf(const Ray &ray) const {
    DCHECK(!_objects.empty());
    float val = 0;
    for (const auto &obj : _objects) {
        val += obj->pdf(ray);
    }
    return val / _objects.size();
}

AABB ObjectGroup::bounding_box() const {
    AABB bbox;
    if (!_bvh.empty()) {
        bbox = _bvh.bounding_box();
    } else {
        bbox = AABB({INF, INF, INF}, {-INF, -INF, -INF});
        for (const auto &obj : _objects) {
            bbox.extend(obj->bounding_box());
        }
    }
    return _mapping.map_bounding_box(bbox);
}

std::vector<Object *> ObjectGroup::children() const {
    std::vector<Object *> child_objs;
    child_objs.reserve(_objects.size());
    for (const auto &obj : _objects) {
        child_objs.emplace_back(obj.get());
    }
    return child_objs;
}

template <bool ENABLE_SURFACE>
void ObjectGroup::intersect_impl(const Ray &ray, Hit &hit, RandEngine *rng) const {
    auto inv_ray = _mapping.map_ray_inverse(ray);
    if (!_bvh.empty()) {
        if constexpr (ENABLE_SURFACE) {
            _bvh.intersect(inv_ray, hit, *rng);
        } else {
            _bvh.intersect_t(inv_ray, hit);
        }
    } else {
        for (const auto &obj : _objects) {
            if constexpr (ENABLE_SURFACE) {
                obj->intersect(inv_ray, hit, *rng);
            } else {
                obj->intersect_t(inv_ray, hit);
            }
        }
    }
    if constexpr (ENABLE_SURFACE) {
        hit.normal = _mapping.map_direction(hit.normal);
        hit.shade_normal = _mapping.map_direction(hit.shade_normal);
    }
}

static std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](char c) { return std::tolower(c); });
    return s;
}

static fs::path get_tex_path(const fs::path &obj_dir, std::string texname) {
    // clean separator
    std::replace(texname.begin(), texname.end(), '\\', '/');
    // do exact match
    auto regular_tex_path = obj_dir / texname;
    if (fs::is_regular_file(regular_tex_path)) {
        return regular_tex_path;
    }
    // if not found, do case-insensitive match
    auto lower_texname = lower(texname);
    for (const auto &entry : fs::directory_iterator(obj_dir)) {
        if (lower(entry.path().filename()) == lower_texname) {
            return entry.path();
        }
    }
    TINYPT_THROW << "Cannot find file " << texname << " in directory " << obj_dir;
}

static Vec3f yup2zup(const Vec3f &v) { return {v.x(), -v.z(), v.y()}; }

class ImageCache {
  public:
    RGBAImage open(const std::string &path, bool grayscale) {
        auto cache_it = _cache.find(std::make_tuple(path, grayscale));
        if (cache_it != _cache.end()) {
            return cache_it->second;
        }
        auto rgba = Image::open(path, grayscale);
        _cache[std::make_tuple(path, grayscale)] = rgba;
        return rgba;
    }

  private:
    std::map<std::tuple<std::string, bool>, RGBAImage> _cache;
};

std::shared_ptr<TriangleMesh> TriangleMesh::from_obj(const std::string &path,
                                                     const std::shared_ptr<Material> &material) {
    tinyobj::ObjReader reader;
    if (!reader.ParseFromFile(path)) {
        LOG_IF(ERROR, !reader.Error().empty()) << "[tinyobj::ObjReader] " << reader.Error();
        TINYPT_THROW << "Failed to parse obj file [" << path << "]";
    }

    auto mesh = std::make_shared<TriangleMesh>();

    LOG_IF(WARNING, !reader.Warning().empty()) << "[tinyobj::ObjReader] " << reader.Warning();

    fs::path obj_dir = fs::path(path).parent_path();

    const auto &attrib = reader.GetAttrib();
    const auto &shapes = reader.GetShapes();

    ImageCache img_cache;

    // build materials
    for (const auto &obj_mat : reader.GetMaterials()) {
        Material mat;
        Vec3f diffuse_color(obj_mat.diffuse);
        Vec3f specular_color(obj_mat.specular);

        RGBTexture emission_texture;
        if (!obj_mat.emissive_texname.empty()) {
            auto emissive_img = img_cache.open(get_tex_path(obj_dir, obj_mat.emissive_texname), false).rgb();
            emission_texture = RGBTexture(std::move(emissive_img));
        } else {
            Vec3f emission_color(obj_mat.emission);
            emission_texture = RGBTexture(emission_color);
        }

        if (obj_mat.illum == 3 || obj_mat.illum == 5) {
            mat = Material(std::make_shared<Metal>(specular_color), std::move(emission_texture));
        } else if (obj_mat.illum == 7) {
            mat =
                Material(std::make_shared<Dielectric>((Vec3f)Vec3f::Ones(), obj_mat.ior), std::move(emission_texture));
        } else {
            AlphaTexture alpha_texture;
            if (!obj_mat.alpha_texname.empty()) {
                auto alpha_map = img_cache.open(get_tex_path(obj_dir, obj_mat.alpha_texname), true).alpha();
                alpha_texture = AlphaTexture(std::move(alpha_map));
            } else if (!is_close(obj_mat.dissolve, 1.f)) {
                alpha_texture = AlphaTexture(obj_mat.dissolve);
            }

            RGBTexture diffuse_texture;
            if (!obj_mat.diffuse_texname.empty()) {
                auto diffuse_rgba = img_cache.open(get_tex_path(obj_dir, obj_mat.diffuse_texname), false);
                if (!diffuse_rgba.alpha().empty() && alpha_texture.empty()) {
                    alpha_texture = AlphaTexture(std::move(diffuse_rgba.alpha()));
                }
                diffuse_texture = RGBTexture(std::move(diffuse_rgba.rgb()));
            } else {
                diffuse_texture = RGBTexture(diffuse_color);
            }

            RGBTexture specular_texture;
            if (!obj_mat.specular_texname.empty()) {
                auto specular_map = img_cache.open(get_tex_path(obj_dir, obj_mat.specular_texname), false).rgb();
                specular_texture = RGBTexture(std::move(specular_map));
            } else if (!specular_color.isZero()) {
                specular_texture = RGBTexture(specular_color);
            }

            BumpTexture bump_texture;
            if (!obj_mat.bump_texname.empty()) {
                auto bump_map = img_cache.open(get_tex_path(obj_dir, obj_mat.bump_texname), true).alpha();
                bump_texture = BumpTexture(std::move(bump_map));
            }

            if (obj_mat.shininess >= EPS && !specular_texture.empty()) {
                constexpr float metal_thresh = 1024;
                auto diffuse = std::make_shared<Lambertian>(std::move(diffuse_texture));
                std::shared_ptr<BXDF> specular;
                if (obj_mat.shininess < metal_thresh) {
                    specular = std::make_shared<Glossy>(obj_mat.shininess, std::move(specular_texture));
                } else {
                    specular = std::make_shared<Metal>(std::move(specular_texture));
                }
                mat = Material(std::make_shared<AddBXDF>(std::move(diffuse), std::move(specular)),
                               std::move(emission_texture), std::move(alpha_texture), std::move(bump_texture));
            } else {
                mat = Material(std::make_shared<Lambertian>(std::move(diffuse_texture)), std::move(emission_texture),
                               std::move(alpha_texture), std::move(bump_texture));
            }
        }
        mesh->_materials.emplace_back(std::move(mat));
    }
    // append default materials at the end
    Material default_material = material ? *material : Material(std::make_shared<Lambertian>((Vec3f)Vec3f::Ones()));
    mesh->_materials.emplace_back(std::move(default_material));

    // build vertices
    TINYPT_CHECK(attrib.vertices.size() % 3 == 0) << "invalid vertex size " << attrib.vertices.size();
    for (size_t i = 0; i < attrib.vertices.size(); i += 3) {
        Vec3f vertex(attrib.vertices[i], attrib.vertices[i + 1], attrib.vertices[i + 2]);
        mesh->_vertices.emplace_back(yup2zup(vertex));
    }

    // build vertex normals
    TINYPT_CHECK(attrib.normals.size() % 3 == 0) << "invalid normal size " << attrib.normals.size();
    for (size_t i = 0; i < attrib.normals.size(); i += 3) {
        Vec3f normal(attrib.normals[i], attrib.normals[i + 1], attrib.normals[i + 2]);
        mesh->_vertex_normals.emplace_back(yup2zup(normal));
    }

    // build texture coordinates
    TINYPT_CHECK(attrib.texcoords.size() % 2 == 0) << "invalid texcoords size " << attrib.texcoords.size();
    for (size_t i = 0; i < attrib.texcoords.size(); i += 2) {
        mesh->_texture_coords.emplace_back(attrib.texcoords[i], attrib.texcoords[i + 1]);
    }

    // build material ids for each face
    size_t num_faces = std::accumulate(shapes.begin(), shapes.end(), 0, [](size_t s, const tinyobj::shape_t &shape) {
        return s + shape.mesh.material_ids.size();
    });
    mesh->_material_ids.reserve(num_faces);
    for (const auto &shape : shapes) {
        for (int obj_mat_id : shape.mesh.material_ids) {
            uint32_t material_id = (obj_mat_id >= 0) ? obj_mat_id : mesh->_materials.size() - 1;
            mesh->_material_ids.emplace_back(material_id);
        }
    }

    // loop over shapes
    std::vector<std::shared_ptr<Object>> groups;
    for (const auto &shape : shapes) {
        std::vector<std::shared_ptr<Object>> faces;
        faces.reserve(shape.mesh.num_face_vertices.size());
        // loop over faces (polygon)
        size_t index_offset = 0;
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++) {
            size_t fv = shape.mesh.num_face_vertices[f];
            TINYPT_CHECK(fv == 3) << "invalid face with " << fv << " vertices";
            std::vector<Vec3f> face_vertices(fv);
            std::array<VertexInfo, 3> face_index{};
            // loop over vertices in the face.
            for (size_t v = 0; v < fv; v++) {
                // access to vertex
                tinyobj::index_t idx = shape.mesh.indices[index_offset + v];
                face_vertices[v] = mesh->_vertices[idx.vertex_index];
                face_index[v] = VertexInfo(idx.vertex_index, idx.normal_index, idx.texcoord_index);
            }
            index_offset += fv;

            faces.emplace_back(std::make_shared<TriangleMeshFace>(face_vertices[0], face_vertices[1], face_vertices[2],
                                                                  mesh->_faces_index.size(), mesh.get()));
            mesh->_faces_index.emplace_back(face_index);
        }
        auto group = std::make_shared<ObjectGroup>(std::move(faces), 4);
        groups.emplace_back(std::move(group));
    }
    mesh->_obj = ObjectGroup(std::move(groups));
    return mesh;
}

} // namespace cpu
} // namespace tinypt
