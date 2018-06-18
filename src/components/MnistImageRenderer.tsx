import { Tensor } from '@tensorflow/tfjs'
import * as React from 'react'
import { PredictionProps } from './Trainer'

export class MnistImageRenderer extends React.Component<PredictionProps<Float32Array, number>> {
  private canvas: HTMLCanvasElement

  public static preProcessInput (tensor: Tensor): Float32Array {
    return tensor.flatten().dataSync() as Float32Array
  }

  public static preProcessOutput (tensor: Tensor): number {
    return (tensor.argMax(1).dataSync() as Float32Array)[0]
  }

  public componentDidMount () {
    this.draw()
  }

  public componentDidUpdate () {
    this.draw()
  }

  public render () {
    const { output, expected } = this.props

    return (
      <div className='Predictor-item'>
        <div
          className={`Predictor-item-label ${output === expected
            ? 'Predictor-item-label_success'
            : 'Predictor-item-label_fail'}`}
        >
          Actual: {expected} | Guess: {output}
        </div>

        <canvas ref={this.saveRef} />
      </div>
    )
  }

  private saveRef = (ref: HTMLCanvasElement) => {
    this.canvas = ref
  }

  private draw = () => {
    const { input: data } = this.props

    const [ width, height ] = [ 28, 28 ]
    this.canvas.width = width
    this.canvas.height = height
    const ctx = this.canvas.getContext('2d')

    if (ctx) {
      const imageData = new ImageData(width, height)
      for (let i = 0; i < height * width; ++i) {
        const j = i * 4
        imageData.data[j + 0] = data[i] * 255
        imageData.data[j + 1] = data[i] * 255
        imageData.data[j + 2] = data[i] * 255
        imageData.data[j + 3] = 255
      }

      ctx.putImageData(imageData, 0, 0)
    }
  }
}
