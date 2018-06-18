import * as React from 'react'

import { MnistDataset } from '../datasets/MnistDataset'
import { MnistModel, MnistModelProps } from '../models/MnistModel'
import { TrainerStore } from '../stores/TrainerStore'
import { MnistImageRenderer } from '../components/MnistImageRenderer'
import { Trainer } from '../components/Trainer'

const mnistModel = new MnistModel()
const mnistDataset = new MnistDataset()
const mnistStore = new TrainerStore<MnistModelProps, Float32Array, number>(mnistModel, mnistDataset)

type MnistTrainerCtor = new () => Trainer<MnistModelProps, Float32Array, number>
const MnistTrainer = Trainer as MnistTrainerCtor

export const Mnist = () => (
  <MnistTrainer store={mnistStore} predictionComponent={MnistImageRenderer} />
)
