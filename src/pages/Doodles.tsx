import * as React from 'react'

import { GoogleDoodlesDataset } from '../datasets/GoogleDoodlesDataset'
import { DoodlesModel, DoodlesModelProps } from '../models/DoodlesModel'
import { TrainerStore } from '../stores/TrainerStore'
import { MnistImageRenderer } from '../components/MnistImageRenderer'
import { Trainer } from '../components/Trainer'

const doodleModel = new DoodlesModel()
const doodleDataset = new GoogleDoodlesDataset()
const doodleStore = new TrainerStore<DoodlesModelProps, Float32Array, number>(doodleModel, doodleDataset)

type DoodleTrainerCtor = new () => Trainer<DoodlesModelProps, Float32Array, number>
const DoodleTrainer = Trainer as DoodleTrainerCtor

export const Doodles = () => (
  <DoodleTrainer store={doodleStore} predictionComponent={MnistImageRenderer} />
)
