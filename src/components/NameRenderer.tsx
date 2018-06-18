import { Tensor, Rank } from '@tensorflow/tfjs'
import * as React from 'react'
import { PredictionProps } from './Trainer'

export class NameRenderer extends React.Component<PredictionProps<string, number>> {
  public static preProcessInput (tensor: Tensor<Rank>) {
    return String.fromCharCode(...tensor.flatten().dataSync())
  }

  public static preProcessOutput (tensor: Tensor): number {
    return (tensor.argMax(1).dataSync() as Float32Array)[0]
  }

  public render () {
    const { input, output, expected } = this.props

    return (
      <div className='Predictor-item'>
        <div
          className={`Predictor-item-label ${output === expected
            ? 'Predictor-item-label_success'
            : 'Predictor-item-label_fail'}`}
        >
          Actual: {this.toLabel(expected)} | Guess: {this.toLabel(output)}
        </div>
        <p>{input}</p>
      </div>
    )
  }

  private toLabel = (i: number) => {
    return i === 0 ? 'M' : 'F'
  }
}
