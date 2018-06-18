import * as React from 'react'
import * as ReactDOM from 'react-dom'
import { initializeIcons } from '@uifabric/icons'
import { App } from './components/App'
import './index.css'
import registerServiceWorker from './registerServiceWorker'

initializeIcons()

ReactDOM.render(
  <App />,
  document.getElementById('root') as HTMLElement
)
registerServiceWorker()
