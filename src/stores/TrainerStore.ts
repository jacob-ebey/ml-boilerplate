import { Model as TfModel, tidy, nextFrame, io, loadModel } from '@tensorflow/tfjs'
import { action, observable } from 'mobx'

import { Vector2 } from '../types/graph'
import { Dataset, Model, PredictResult } from '../types/models'
import { ModelPropsStore } from './ModelPropsStore'
import { TrainerPropsStore, TrainerStoreProps } from './TrainerPropsStore'

export const defaultTrainerProps: TrainerStoreProps = {
  batchSize: 64,
  batches: 100,
  testBatchSize: 1000,
  testFrequency: 100
}

export class TrainerStore<TModelProps, TInput, TResult> {
  @observable public error: string | null = null
  @observable public loadingDataset: boolean = false
  @observable public loadedDataset: boolean = false
  @observable public training: boolean = false
  @observable public props: TrainerPropsStore
  @observable public modelProps: ModelPropsStore<TModelProps>
  @observable public loss: Vector2[] = []
  @observable public accuracy: Vector2[] = []
  @observable public model: Model<TModelProps>
  @observable public dataset: Dataset
  @observable public tfModel: TfModel | null = null
  @observable public savedModels: string[] = []
  @observable public predictionResults: Array<PredictResult<TInput, TResult>> = []

  private iteration: number = 0

  public constructor (
    model: Model<TModelProps>,
    dataset: Dataset,
    defaultProps: TrainerStoreProps = defaultTrainerProps
  ) {
    this.model = model
    this.dataset = dataset
    this.props = new TrainerPropsStore(defaultProps)
    this.modelProps = new ModelPropsStore(model.config)
  }

  @action
  public loadDataset = () => {
    new Promise(async (resolve, reject) => {
      this.savedModels = Object.getOwnPropertyNames(await io.listModels())

      if (this.loadedDataset || this.loadingDataset) {
        return
      }

      this.loadingDataset = true

      try {
        await this.dataset.load()
        this.loadedDataset = true
      } catch (err) {
        reject(err)
      } finally {
        this.loadingDataset = false
        resolve()
      }
    }).catch((err) => {
      this.error = 'Failed to load dataset'
      console.error(err)
    })
  }

  @action
  public resetModel = () => {
    this.tfModel = null

    this.iteration = 0
    this.loss = []
    this.accuracy = []
  }

  @action
  public saveModel = () => {
    new Promise(async (resolve, reject) => {
      try {
        if (this.tfModel) {
          const rightNow = new Date()
          const timestamp = rightNow.toISOString().slice(0, 10)
            + `-${rightNow.getHours()}:${rightNow.getMinutes()}:${rightNow.getSeconds()}`

          await this.tfModel.save(`localstorage://${this.model.label}-${timestamp}`)

          this.savedModels = Object.getOwnPropertyNames(await io.listModels())
          resolve()
        }
      } catch (err) {
        reject(err)
      }
    }).catch((err) => {
      this.error = 'Failed to save model'
      console.error(err)
    })
  }

  public loadModel = (model: string) => {
    return () => {
      new Promise(async (resolve, reject) => {
        try {
          this.resetModel()
          this.tfModel = await loadModel(model)
          this.model.compileModel(this.tfModel, this.modelProps.values)
          resolve()
        } catch (err) {
          reject(err)
        }
      }).catch((err) => {
        this.error = 'Failed to load model'
        console.error(err)
      })
    }
  }

  public deleteModel = (model: string) => {
    return () => {
      new Promise(async (resolve, reject) => {
        try {
          await io.removeModel(model)
          this.savedModels = Object.getOwnPropertyNames(await io.listModels())
          resolve()
        } catch (err) {
          reject(err)
        }
      }).catch((err) => {
        this.error = 'Failed to delete model'
        console.error(err)
      })
    }
  }

  @action
  public train = () => {
    new Promise(async (resolve, reject) => {
      if (this.training) {
        return resolve()
      }

      this.training = true

      if (!this.tfModel) {
        this.tfModel = this.model.createAndCompileModel(this.modelProps.values)
      }

      const props = this.props.values

      try {
        for (let i = 0; i < props.batches; i++) {
          const { batch, validationData } = tidy<any>(() => {
            const newBatch = this.dataset.nextTrainBatch(props.batchSize)

            let newValidationData
            if (i % props.testFrequency === 0) {
              const testBatch = this.dataset.nextTestBatch(props.testBatchSize)
              newValidationData = [
                testBatch.input,
                testBatch.expected
              ]
            }

            return { batch: newBatch, validationData: newValidationData }
          })

          const history = await this.tfModel.fit(
            batch.input,
            batch.expected,
            {
              batchSize: props.batchSize,
              epochs: 1,
              validationData
            }
          )

          const loss = history.history.loss[0]
          const accuracy = history.history.acc[0]

          if (typeof loss === 'number') {
            this.loss = [
              ...this.loss,
              {
                x: this.iteration + i,
                y: loss
              }
            ]
          }

          if (typeof accuracy === 'number') {
            this.accuracy = [
              ...this.accuracy,
              {
                x: this.iteration + i,
                y: accuracy
              }
            ]
          }

          await nextFrame()
        }

        this.iteration += props.batches
      } catch (err) {
        return reject(err)
      } finally {
        this.training = false
        return resolve()
      }
    }).catch((err) => {
      this.error = 'An error occurred while training'
      console.error(err)
    })
  }
}
