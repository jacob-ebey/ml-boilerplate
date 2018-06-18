import { tensor2d } from '@tensorflow/tfjs'
import { Batch, Dataset } from '../types/models'

export class XorDataset implements Dataset {
  public load () {
    return Promise.resolve()
  }

  public nextTrainBatch = (batchSize: number) => {
    return this.getBatch(batchSize)
  }

  public nextTestBatch = (batchSize: number) => {
    return this.getBatch(batchSize)
  }

  private getBatch = (batchSize: number): Batch => {
    const input: number[][] = []
    const expected: number[][] = []

    for (let i = 0; i < batchSize; i++) {
      const a = Math.random() > 0.5 ? 1 : 0
      const b = Math.random() > 0.5 ? 1 : 0

      input.push([ a, b ])
      expected.push([ a ^ b ])
    }

    return {
      input: tensor2d(input, [ batchSize, 2 ]),
      expected: tensor2d(expected, [ batchSize, 1 ])
    }
  }
}
