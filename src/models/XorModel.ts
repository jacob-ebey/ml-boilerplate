import { Model as TfModel, layers, sequential, train } from '@tensorflow/tfjs'

import { Model, ModelProps } from '../types/models'

export interface XorModelProps {
  learningRate: number
  layer1Units: number
}

export class XorModel implements Model<XorModelProps> {
  public get label (): string { return 'XOR DENSE Model' }

  public get config (): ModelProps<XorModelProps> {
    return {
      learningRate: {
        type: 'number',
        label: 'Learning Rate',
        defaultValue: 0.1
      },
      layer1Units: {
        type: 'number',
        label: 'Layer1 (DENSE): Units',
        defaultValue: 3
      }
    }
  }

  public createAndCompileModel = (props: XorModelProps) => {
    const model = sequential()

    model.add(layers.dense({
      inputShape: [ 2 ],
      activation: 'sigmoid',
      kernelInitializer: 'varianceScaling',
      units: props.layer1Units
    }))

    model.add(layers.dense({
      activation: 'sigmoid',
      kernelInitializer: 'varianceScaling',
      units: 1
    }))

    this.compileModel(model, props)

    return model
  }

  public compileModel = (model: TfModel, props: XorModelProps) => {
    const optimizer = train.rmsprop(props.learningRate)

    model.compile({
      loss: 'meanSquaredError',
      metrics: [ 'accuracy' ],
      optimizer
    })
  }
}
