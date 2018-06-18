import * as React from 'react'

import { XorDataset } from '../datasets/XorDataset'
import { XorModel, XorModelProps } from '../models/XorModel'
import { TrainerStore } from '../stores/TrainerStore'
import { XorRenderer } from '../components/XorRenderer'
import { Trainer } from '../components/Trainer'

const xorModel = new XorModel()
const xorDataset = new XorDataset()
const xorStore = new TrainerStore<XorModelProps, number[], number>(xorModel, xorDataset)

type XorTrainerCtor = new () => Trainer<XorModelProps, number[], number>
const XorTrainer = Trainer as XorTrainerCtor

export const Xor = () => (
  <XorTrainer store={xorStore} predictionComponent={XorRenderer} />
)
