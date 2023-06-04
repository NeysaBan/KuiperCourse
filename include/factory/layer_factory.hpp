//
// Created by fss on 22-12-21.
//

#ifndef KUIPER_COURSE_INCLUDE_FACTORY_LAYER_FACTORY_HPP_
#define KUIPER_COURSE_INCLUDE_FACTORY_LAYER_FACTORY_HPP_
#include "ops/op.hpp"
#include "layer/layer.hpp"

namespace kuiper_infer
{
  class LayerRegisterer
  {
  public:
    typedef std::shared_ptr<Layer> (*Creator)(const std::shared_ptr<Operator> &op); // 函数指针？

    typedef std::map<OpType, Creator> CreateRegistry; // 实际上的注册表

    static void RegisterCreator(OpType op_type, const Creator &creator); // 注册算子，key=op，value=layer

    static std::shared_ptr<Layer> CreateLayer(const std::shared_ptr<Operator> &op); // 根据op返回layer

    // typedef std::map<OpType, Creator> CreateRegistry;
    static CreateRegistry &Registry(); // 创建注册表的方法
  };

  class LayerRegistererWrapper
  {
  public:
    // 注册包装类的构造函数
    LayerRegistererWrapper(OpType op_type, const LayerRegisterer::Creator &creator)
    {
      LayerRegisterer::RegisterCreator(op_type, creator);
    }
  };

}
#endif // KUIPER_COURSE_INCLUDE_FACTORY_LAYER_FACTORY_HPP_
