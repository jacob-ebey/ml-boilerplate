import { Tensor, Rank, tidy } from '@tensorflow/tfjs'
import * as React from 'react'
import { observer } from 'mobx-react'
import { PrimaryButton, DefaultButton } from 'office-ui-fabric-react/lib/Button'
import { IContextualMenuItem } from 'office-ui-fabric-react/lib/ContextualMenu'
import { TextField } from 'office-ui-fabric-react/lib/TextField'
import { Panel, PanelType } from 'office-ui-fabric-react/lib/Panel'
import { Spinner, SpinnerSize } from 'office-ui-fabric-react/lib/Spinner'
import { LineChart } from 'react-easy-chart'

import { TrainerStore } from '../stores/TrainerStore'
import { PredictResult, PropConfig } from '../types/models'

import './Trainer.css'

export interface PredictionProps<TInput, TResult> {
  input: TInput
  output: TResult
  expected: TResult
}

export interface TrainerProps<TModelProps, TInput, TResult> {
  store: TrainerStore<TModelProps, TInput, TResult>
  predictionComponent: React.ComponentType<PredictionProps<TInput, TResult>>
}

@observer
export class Trainer<TModelProps, TInput, TResult> extends React.Component<TrainerProps<TModelProps, TInput, TResult>> {
  public componentWillMount () {
    this.props.store.loadDataset()
  }

  public render () {
    const {
      error,
      loadingDataset,
      training,
      accuracy,
      loss,
      model: { config },
      props,
      modelProps,
      savedModels,
      predictionResults
    } = this.props.store

    return (
      <React.Fragment>
        <div className='Trainer'>
          <div className='Trainer-content'>
            {
              loadingDataset || !savedModels ? (
                <Spinner className='Trainer-loader' size={SpinnerSize.large} label='Loading Dataset...' />
              ) : (
                <React.Fragment>
                  <div className='Trainer-graphs'>
                    <div>
                      <p>Loss</p>
                      <LineChart
                        axes={true}
                        data={[ loss ]}
                        width={350}
                        height={200}
                      />
                    </div>
                    <div>
                      <p>Accuracy</p>
                      <LineChart
                        axes={true}
                        data={[ accuracy ]}
                        width={350}
                        height={200}
                      />
                    </div>
                  </div>
                  <div className='Predictor-items'>
                    {
                      predictionResults.map((result, i) => (
                        <this.props.predictionComponent key={i} {...result} />
                      ))
                    }
                  </div>
                </React.Fragment>
              )
            }

            {
              error && (
                <div className='ms-bgColor-redDark ms-fontColor-white Trainer-message'>{error}</div>
              )
            }

            {
              training && (
                <div className='ms-bgColor-greenDark ms-fontColor-white Trainer-message'>Training...</div>
              )
            }
          </div>
          <div id='Trainer-props-panel' />
        </div>
        <Panel
          className='Trainer-flex-panel'
          onRenderHeader={this.renderPanelHeader}
          isBlocking={false}
          hasCloseButton={false}
          isOpen={true}
          type={PanelType.medium}
          layerProps={{
            hostId: 'Trainer-props-panel',
            className: 'Trainer-flex-panel-layer'
          }}
        >
          <p className='ms-font-xl'>Training</p>
          <TextField
            label='Batches'
            type='number'
            value={props.batches.toString()}
            onChanged={props.onValueChange('batches')}
          />
          <TextField
            label='Batch Size'
            type='number'
            value={props.batchSize.toString()}
            onChanged={props.onValueChange('batchSize')}
          />
          <TextField
            label='Test Batch Size'
            type='number'
            value={props.testBatchSize.toString()}
            onChanged={props.onValueChange('testBatchSize')}
          />
          <TextField
            label='Test Frequency'
            type='number'
            value={props.testFrequency.toString()}
            onChanged={props.onValueChange('testFrequency')}
          />

          <p className='ms-font-xl'>Model</p>
          {
              Object.getOwnPropertyNames(config).map((key: string) => {
                if (!(key in config)) {
                  return null
                }
                const item = config[key] as PropConfig

                return (
                  <TextField
                    key={key}
                    label={item.label}
                    type={item.type}
                    value={modelProps.getValue(key)}
                    onChanged={modelProps.onValueChange(key)}
                  />
                )
              })
            }
        </Panel>
      </React.Fragment>
    )
  }

  private renderPanelHeader = () => {
    const {
      loadedDataset,
      model: { label },
      train,
      training,
      resetModel,
      saveModel,
      savedModels,
      loadModel,
      deleteModel
    } = this.props.store

    return (
      <div className='Trainer-panel-header'>
        <p className='ms-font-xxl'>{label}</p>
        <DefaultButton
          text='Reset'
          onClick={resetModel}
          split={savedModels.length > 0}
          menuProps={savedModels.length > 0 ? {
            items: savedModels.map((model): IContextualMenuItem => ({
              key: model,
              text: model,
              onClick: loadModel(model),
              split: true,
              subMenuProps: {
                items: [ {
                  key: 'delete',
                  text: 'Delete',
                  onClick: deleteModel(model)
                } ]
              }
            }))
          } : undefined}
        />
        <DefaultButton text='Save' onClick={saveModel} />
        <br /><br />
        <PrimaryButton text='Train' disabled={!loadedDataset || training} onClick={train} />
        <DefaultButton text='Predict' disabled={!loadedDataset} onClick={this.predict} />
      </div>
    )
  }

  private predict = () => {
    const {
      predictionComponent,
      store: {
        dataset,
        model,
        modelProps
      }
     } = this.props

    if (!this.props.store.tfModel) {
      this.props.store.tfModel = model.createAndCompileModel(modelProps.values)
    }

    const examples = 20

    const batch = dataset.nextTestBatch(examples)

    const processInput = (tempInput: Tensor) => (predictionComponent as any).preProcessInput
        ? (predictionComponent as any).preProcessInput(tempInput)
        : tempInput

    const processOutput = (tempOutput: Tensor) => (predictionComponent as any).preProcessOutput
        ? (predictionComponent as any).preProcessOutput(tempOutput)
        : tempOutput

    tidy(() => {
      if (!this.props.store.tfModel) {
        this.props.store.error = 'Failed to initialize model for prediction'
        return
      }

      const output = this.props.store.tfModel.predict(batch.input) as Tensor<Rank>

      if (batch.expected.shape[1] === output.shape[1]) {
        const results: Array<PredictResult<TInput, TResult>> = []

        console.log(output.shape)

        for (let i = 0; i < output.shape[0]; i++) {
          results.push({
            input: processInput(batch.input.slice([ i, 0 ], [ 1, batch.input.shape[1] ])),
            expected: processOutput(batch.expected.slice([ i, 0 ], [ 1, batch.expected.shape[1] ])),
            output: processOutput(output.slice([ i, 0 ], [ 1, output.shape[1] ]))
          })
        }

        this.props.store.predictionResults = results
      }
    })
  }
}
