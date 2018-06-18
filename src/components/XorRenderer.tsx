import { Tensor, Rank } from '@tensorflow/tfjs'
import * as React from 'react'
import { PredictionProps } from './Trainer'

export class XorRenderer extends React.Component<PredictionProps<number[], number>> {
  public static preProcessInput (tensor: Tensor<Rank>) {
    return (tensor.dataSync() as Float32Array)
  }

  public static preProcessOutput (tensor: Tensor): number {
    return (tensor.dataSync() as Float32Array)[0]
  }

  public render () {
    const { input, output, expected } = this.props

    return (
      <div className='Predictor-item'>
        <div
          className={`Predictor-item-label ${output > 0.5 === !!expected
            ? 'Predictor-item-label_success'
            : 'Predictor-item-label_fail'}`}
        >
          Actual: {expected} | Guess: {output}
        </div>
        <p>{input[0]} ^ {input[1]} = {expected}</p>
      </div>
    )
  }
}
