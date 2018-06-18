import { decorate, observable } from 'mobx'

import { ModelProps, PropConfig } from '../types/models'

export class ModelPropsStore<TModelProps extends {}> {
  private props: ModelProps<TModelProps>
  private cache: any = {}

  public constructor (props: ModelProps<TModelProps>) {
    this.props = props

    const toDecorate: any = {}
    Object.getOwnPropertyNames(props).forEach((key: string) => {
      if (!(key in props)) {
        return
      }

      const config = props[key] as PropConfig

      this.cache[key] = config.defaultValue.toString()
      toDecorate[key] = observable
    })

    decorate(this.cache, toDecorate)
  }

  public get values (): TModelProps {
    const result: any = {}

    Object.getOwnPropertyNames(this.props).forEach((key: string) => {
      if (!(key in this.props)) {
        return
      }

      result[key] = (this.props[key] as PropConfig).type === 'number'
        ? Number.parseFloat(this.cache[key])
        : !!this.cache[key]
    })

    return result
  }

  public getValue = (key: string) => {
    return this.cache[key]
  }

  public onValueChange = (key: string) => (value: string) => {
    this.cache[key] = value
  }
}
