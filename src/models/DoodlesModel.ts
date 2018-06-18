import { Model as TfModel, layers, sequential, train } from '@tensorflow/tfjs'

import { Model, ModelProps } from '../types/models'

// Define your user configurable properties
export interface DoodlesModelProps {
  learningRate: number
  layer1Filters: number
  layer1KernelSize: number
  layer2PoolSize: number
  layer2Strides: number
  layer3Filters: number
  layer3KernelSize: number
  layer4PoolSize: number
  layer4Strides: number
}

export class DoodlesModel implements Model<DoodlesModelProps> {
  public get label (): string { return 'Doodles CNN Model' }

  // Give your user configurable properties default values
  public get config (): ModelProps<DoodlesModelProps> {
    return {
      learningRate: {
        type: 'number',
        label: 'Learning Rate',
        defaultValue: 0.15
      },
      layer1Filters: {
        type: 'number',
        label: 'Layer1 (Cov2D): Filters',
        defaultValue: 8
      },
      layer1KernelSize: {
        type: 'number',
        label: 'Layer1 (Cov2D): Kernel Size',
        defaultValue: 5
      },
      layer2PoolSize: {
        type: 'number',
        label: 'Layer2 (MaxPool2D): Pool Size',
        defaultValue: 2
      },
      layer2Strides: {
        type: 'number',
        label: 'Layer2 (MaxPool2D): ',
        defaultValue: 2
      },
      layer3Filters: {
        type: 'number',
        label: 'Layer3 (Cov2D): Filters',
        defaultValue: 8
      },
      layer3KernelSize: {
        type: 'number',
        label: 'Layer3 (Cov2D): Kernel Size',
        defaultValue: 5
      },
      layer4PoolSize: {
        type: 'number',
        label: 'Layer4 (MaxPool2D): Pool Size',
        defaultValue: 2
      },
      layer4Strides: {
        type: 'number',
        label: 'Layer4 (MaxPool2D): ',
        defaultValue: 2
      }
    }
  }

  // Build your model
  public createAndCompileModel = (props: DoodlesModelProps) => {
    const model = sequential()

    model.add(layers.conv2d({
      inputShape: [ 28, 28, 1 ],
      kernelSize: props.layer1KernelSize,
      filters: props.layer1Filters,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'VarianceScaling'
    }))

    model.add(layers.maxPooling2d({
      poolSize: [ props.layer2PoolSize, props.layer2PoolSize ],
      strides: [ props.layer2Strides, props.layer2Strides ]
    }))

    model.add(layers.conv2d({
      kernelSize: props.layer3KernelSize,
      filters: props.layer3Filters,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'VarianceScaling'
    }))

    model.add(layers.maxPooling2d({
      poolSize: [ props.layer4PoolSize, props.layer4PoolSize ],
      strides: [ props.layer4Strides, props.layer4Strides ]
    }))

    model.add(layers.flatten())

    model.add(layers.dense({
      units: 3,
      kernelInitializer: 'VarianceScaling',
      activation: 'softmax'
    }))

    // Reuse the compileModel method to keep the two separate paths (creating and re-initializing)
    // in sync :)
    this.compileModel(model, props)

    return model
  }

  // Compile the model with an optimizer for training
  public compileModel = (model: TfModel, props: DoodlesModelProps) => {
    const optimizer = train.adam(props.learningRate)

    model.compile({
      loss: 'categoricalCrossentropy',
      metrics: [ 'accuracy' ],
      optimizer
    })
  }
}
