#include <tinypt/tinypt.h>

class SceneFactory {
  public:
    SceneFactory();

    tinypt::Scene make_scene(const std::string &name) const;

  private:
    void register_scene_builder(const std::string &name, std::function<tinypt::Scene()> scene_fn);

    static tinypt::Scene make_cornell_sphere();
    static tinypt::Scene make_cornell_box();
    static tinypt::Scene make_breakfast_room();
    static tinypt::Scene make_living_room();
    static tinypt::Scene make_fireplace_room();
    static tinypt::Scene make_rungholt();
    static tinypt::Scene make_dabrovic_sponza();
    static tinypt::Scene make_salle_de_bain();

    static tinypt::Scene make_debug();
    static tinypt::Scene make_debug2();

  private:
    std::unordered_map<std::string, std::function<tinypt::Scene()>> _scene_builders;
};
