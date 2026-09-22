# Third-party components

ModelPort's original source is licensed under Apache-2.0. Dependencies and model
weights have their own licenses. ModelPort does not grant rights to third-party models.

The dashboard bundles React and React DOM (MIT) and Lucide icons (ISC; includes
Feather-derived icons under MIT). Their full notices are included in
`web/dist/third-party-licenses.txt`, also packaged in the wheel's dashboard assets.

Python runtime dependencies are installed separately from the wheel. Their wheels
include their license notices. The Docker image retains those installed notices.
The release includes `dependency-inventory.json` with the installed Python package
versions and package-reported license metadata, plus the locked JavaScript packages.
This inventory is a record of the build environment, not a license compatibility opinion.

The public model examples contain revision IDs and checksums only. No MNIST or BERT
weights are redistributed. Review each model repository's terms before downloading
or using its files. The tiny random BERT example tests mechanics, not model quality.
