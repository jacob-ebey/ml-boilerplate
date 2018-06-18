import { observable } from 'mobx'

export interface TrainerStoreProps {
  batchSize: number
  batches: number
  testBatchSize: number
  testFrequency: number
}

export class TrainerPropsStore {
  @observable public batches: string
  @observable public batchSize: string
  @observable public testBatchSize: string
  @observable public testFrequency: string

  public constructor (initialValues: TrainerStoreProps) {
    this.batches = initialValues.batches.toString()
    this.batchSize = initialValues.batchSize.toString()
    this.testBatchSize = initialValues.testBatchSize.toString()
    this.testFrequency = initialValues.testFrequency.toString()
  }

  public get values (): TrainerStoreProps {
    return {
      batches: Number.parseFloat(this.batches),
      batchSize: Number.parseFloat(this.batchSize),
      testBatchSize: Number.parseFloat(this.testBatchSize),
      testFrequency: Number.parseFloat(this.testFrequency)
    }
  }

  public onValueChange = (prop: keyof TrainerStoreProps) => (value: string) => {
    this[prop] = value
  }
}
