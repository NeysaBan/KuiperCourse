//
// Created by fss on 22-12-21.
//

#include "factory/layer_factory.hpp"
#include <glog/logging.h>

namespace kuiper_infer
{
  // 单例设计编程模式
  //  全局当中有且只有一个变量
  //  任意次和任意一方去调用都会得到这个唯一的变量
  // 这里的唯一变量是全局的注册表 存的时候是这个，取得时候也需要是这个

  void LayerRegisterer::RegisterCreator(OpType op_type, const Creator &creator)
  {
    CHECK(creator != nullptr) << "Layer creator is empty";
    CreateRegistry &registry = Registry();
    CHECK_EQ(registry.count(op_type), 0) << "Layer type: " << int(op_type) << " has already registered!";
    registry.insert({op_type, creator});
  }

  std::shared_ptr<Layer> LayerRegisterer::CreateLayer(const std::shared_ptr<Operator> &op)
  {
    CreateRegistry &registry = Registry();
    const OpType op_type = op->op_type_;

    LOG_IF(FATAL, registry.count(op_type) <= 0) << "Can not find the layer type: " << int(op_type);
    const auto &creator = registry.find(op_type)->second;

    LOG_IF(FATAL, !creator) << "Layer creator is empty!";
    /*
    这里调用的应该是
    std::shared_ptr<Layer> ReluLayer::CreateInstance(const std::shared_ptr<Operator> &op) {
      std::shared_ptr<Layer> relu_layer = std::make_shared<ReluLayer>(op);
      return relu_layer;
    }
    */
    std::shared_ptr<Layer> layer = creator(op);
    LOG_IF(FATAL, !layer) << "Layer init failed!";
    return layer;
  }

  // 注册的是一个初始化方法

  LayerRegisterer::CreateRegistry &LayerRegisterer::Registry()
  {
    // 使用static
    // 只会被初始化一次，后面返回的地址都是相同的
    // 简单来说第一次，调用的时候 new CreateRegistry 存放到一个kRegistry (static)
    // 后续调用的时候，只会返回kRegistry (static)
    static CreateRegistry *kRegistry = new CreateRegistry();
    CHECK(kRegistry != nullptr) << "Global layer register init failed!";
    return *kRegistry;
  }
}
