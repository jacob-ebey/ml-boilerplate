import * as React from 'react'
import { BrowserRouter, Route, Link } from 'react-router-dom'

import { Fabric } from 'office-ui-fabric-react/lib/Fabric'
import 'office-ui-fabric-react/dist/css/fabric.min.css'

import { Home } from '../pages/Home'
import { Mnist } from '../pages/Mnist'
import { Xor } from '../pages/Xor'

import './App.css'

export class App extends React.Component {
  public render () {
    return (
      <BrowserRouter>
        <Fabric>
          <div className='App'>
            <header className='App-header'>
              <h2 className='App-title'><Link to='/' className='App-title'>Tensorflow JS Boilerplate</Link></h2>
              <div className='App-header-links'>
                <Link to='/mnist' className='App-header-link'>MNIST</Link>
                <Link to='/xor' className='App-header-link'>XOR</Link>
              </div>
            </header>
            <div className='App-content'>
                <Route exact path='/' component={Home} />
                <Route path='/mnist' component={Mnist} />
                <Route path='/xor' component={Xor} />
            </div>
          </div>
        </Fabric>
      </BrowserRouter>
    )
  }
}
