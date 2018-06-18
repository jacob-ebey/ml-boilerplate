import { Model as TfModel, Tensor } from '@tensorflow/tfjs'

export type ModelPropType = 'number' | 'boolean'

export interface BasePropConfig<T> {
  label: string
  type: ModelPropType
  defaultValue: T
}

export interface BooleanPropConfig extends BasePropConfig<boolean> {
  type: 'boolean'
}

export interface NumberPropConfig extends BasePropConfig<number> {
  type: 'number'
}

export type PropConfig = BooleanPropConfig | NumberPropConfig

export type ModelProps<TModelProps> = {
  [key in keyof TModelProps]: PropConfig
}

export interface Model<TModelProps> {
  readonly label: string
  readonly config: ModelProps<TModelProps>
  createAndCompileModel (props: TModelProps): TfModel
  compileModel (model: TfModel, props: TModelProps): void
}

export interface Batch {
  input: Tensor
  expected: Tensor
}

export interface PredictResult<TInput, TResult> {
  input: TInput
  output: TResult
  expected: TResult
}

export interface Dataset {
  load (): Promise<void>
  nextTrainBatch (batchSize: number): Batch
  nextTestBatch (batchSize: number): Batch
}
