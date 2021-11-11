# dl21
Deep Learning FS21 Project

---
## TODO's
- todo

---
## Roadmap
- todo

---
##Structure
- utils
  - _image_operations.py:_
  - _get_weights.py:_digit
classification networks
- 

---
## FairML
_**Perturbation:**_ The basic idea behind FairML (and many other attempts to audit or interpret model behavior) is to measure a model’s 
dependence on its inputs by changing them.  If a small change to an input feature dramatically changes the output, the model is sensitive to the feature.

Therefore, if a model places high importance on a specific feature, then a slight change (perturbation) would result in a big change to the prediction (that is the assumption).

__But what if the input attributes are correlated?__ Perturbing one feature alone will (in that case) not provide an accurate measure of the model’s dependency on the feature.
One has to perturb the other input attributes as well.

The trick FairML uses to counter this multicollinearity is orthogonal projection. FairML orthogonally projects the input to measure the dependence of the predictive model on each attribute.

---
## Links and References

- FairML Article: https://blog.fastforwardlabs.com/2017/03/09/fairml-auditing-black-box-predictive-models.html
- 
