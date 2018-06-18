import { tensor2d } from '@tensorflow/tfjs'
import axios from 'axios'
import { Dataset } from '../types/models'

export interface GenderDataPoint {
  m: number
  f: number
}

export interface GenderDataPointCol {
  [0]: string
  [1]: GenderDataPoint
}

function shuffle (a: any[]) {
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [ a[i], a[j] ] = [ a[j], a[i] ]
  }

  return a
}

function toCharArray (str: string) {
  const result = []
  const length = str.length

  for (let i = 0; i < length; i++) {
    const code = str.charCodeAt(i)
    result.push(code)
  }

  return result
}

export class GenderDataset implements Dataset {
  private trainData: GenderDataPointCol[]
  private testData: GenderDataPointCol[]
  private trainIndex: number = 0
  private testIndex: number = 0

  public async load () {
    const res = await axios.get('/name_list_no_dups.json')

    const data = shuffle(res.data)

    const splice = parseInt(`${data.length * 0.6}`, 10)
    this.trainData = data.slice(0, splice)
    this.testData = data.slice(splice)
  }

  public nextTrainBatch (batchSize: number) {
    let data = this.trainData.slice(this.trainIndex, this.trainIndex + batchSize)
    this.trainIndex = this.trainIndex + batchSize

    if (data.length < batchSize) {
      this.trainIndex = batchSize - data.length
      data = data.concat(this.trainData.slice(0, this.trainIndex))
    }

    return this.toTensor(data, batchSize)
  }

  public nextTestBatch (batchSize: number) {
    let data = this.testData.slice(this.testIndex, this.testIndex + batchSize)
    this.testIndex = this.testIndex + batchSize

    if (data.length < batchSize) {
      this.testIndex = batchSize - data.length
      data = data.concat(this.testData.slice(0, this.testIndex))
    }

    return this.toTensor(data, batchSize)
  }

  private toTensor (data: GenderDataPointCol[], batchSize: number) {
    const inputs: number[][] = []
    const labels: number[][] = []

    data.forEach((e) => {
      const name = ('                ' + e[0]).slice(-16)
      let input = toCharArray(name)
      input = input.fill(0, input.length, 16)

      inputs.push(input)

      const m = e[1].m > e[1].f ? 1 : 0
      const f = e[1].m > e[1].f ? 0 : 1
      labels.push([ m, f ])
    })

    return {
      input: tensor2d(inputs, [ batchSize, 16 ]),
      expected: tensor2d(labels, [ batchSize, 2 ])
    }
  }
}
