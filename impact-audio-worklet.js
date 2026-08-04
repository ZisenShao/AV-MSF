class ImpactSynthProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const {
      frequencies,
      gains,
      dampings,
      duration,
      outputGain,
    } = options.processorOptions;
    const count = frequencies.length;
    this.real = new Float32Array(gains);
    this.imaginary = new Float32Array(count);
    this.multiplierReal = new Float32Array(count);
    this.multiplierImaginary = new Float32Array(count);
    this.outputGain = outputGain;
    this.samplesRemaining = Math.ceil(duration * sampleRate);
    this.stopped = false;

    for (let mode = 0; mode < count; mode += 1) {
      const decay = Math.exp(-dampings[mode] / sampleRate);
      const omega = (2 * Math.PI * frequencies[mode]) / sampleRate;
      this.multiplierReal[mode] = decay * Math.cos(omega);
      this.multiplierImaginary[mode] = decay * Math.sin(omega);
    }
    this.port.onmessage = (event) => {
      if (event.data?.type === "stop") this.stopped = true;
    };
  }

  process(_inputs, outputs) {
    if (this.stopped) return false;
    const output = outputs[0][0];
    for (let sample = 0; sample < output.length; sample += 1) {
      if (this.samplesRemaining <= 0) {
        this.port.postMessage("ended");
        return false;
      }
      let value = 0;
      for (let mode = 0; mode < this.real.length; mode += 1) {
        const real = this.real[mode];
        const imaginary = this.imaginary[mode];
        value += real;
        this.real[mode] =
          real * this.multiplierReal[mode] -
          imaginary * this.multiplierImaginary[mode];
        this.imaginary[mode] =
          real * this.multiplierImaginary[mode] +
          imaginary * this.multiplierReal[mode];
      }
      output[sample] = value * this.outputGain;
      this.samplesRemaining -= 1;
    }
    return true;
  }
}

registerProcessor("avmsf-impact-synth", ImpactSynthProcessor);
