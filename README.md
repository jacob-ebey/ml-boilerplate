# ML-Boilerplate

A boilerplate for getting started with machine learning in the browser using Tensorflow.JS and React.

If you are looking for an easy way to get started building or training your own models there is no
easier way. Just clone, install dependencies and run as there are no tools required that are not available
via NPM.

## Getting started

Install Yarn if you don't have it
```
> npm install -g yarn
```

Install packages and run in dev mode

```
> yarn
> yarn start
```

## Why use this?

Getting started with deep learning can be a pain in the ass as there are a lot of things to take into account:
How am I going to expose model variables? How am I going to load my training data? How am I going to plot my
loss and validation data? How do I save my models? How can I keep training an existing model? Etc...

This project is an attempt to alleviate some of these concerns for new comers and shame some of the experts
into releasing work that is consumable by your average developer.

## What's going on?

This project exposes 3 main concepts to make your life easier when developing models:

- The interface: ```Model<TModelProps>``` where you implement your model and expose user configurable properties
- The interface ```Dataset``` where you implement the retrieval of your training / test data
- The React component ```Trainer<TModelProps, TInput, TResult>``` for rendering of samples
