# Acknowledgements and source attribution

AEDR-SED builds on the model and training framework from
[EfficientSED](https://github.com/theMoro/EfficientSED) and
[PretrainedSED](https://github.com/fschmid56/PretrainedSED).
The original MIT copyright notice is preserved in `LICENSE`.

The BEATs implementation retains its original Microsoft and fairseq source
comments and attributions. Its internal Transformer is part of the acoustic
encoder and is retained. Alternative acoustic backbones and standalone
temporal comparison decoders are not included in this release.
The encoding and decoding utilities retain references to the DCASE DESED
implementations on which they are based. Consult the source headers for
component-specific attribution.

AudioSet and DESED datasets and pretrained model binaries are obtained
separately from their original providers. No dataset audio, trained checkpoint,
historical metric table or experimental log is distributed in this repository.
