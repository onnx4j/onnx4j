/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.onnx4j.opsets.domain.aiOnnx.v1.ops;

import java.util.List;

import org.onnx4j.Inputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.model.graph.node.attributes.FloatAttribute;
import org.onnx4j.model.graph.node.attributes.IntAttribute;
import org.onnx4j.model.graph.node.attributes.IntsAttribute;
import org.onnx4j.opsets.domain.aiOnnx.v1.AiOnnxOperatorV1;
import org.onnx4j.opsets.operator.Field;
import org.onnx4j.opsets.operator.Field.TypeConstraint;
import org.onnx4j.opsets.operator.OperatorInputs;
import org.onnx4j.opsets.operator.OperatorOutputs;
import org.onnx4j.opsets.operator.fields.AttributeField;
import org.onnx4j.opsets.operator.fields.InputField;
import org.onnx4j.opsets.operator.fields.OutputField;
import org.onnx4j.opsets.operator.output.MultiOperatorOutputs;
import org.onnx4j.tensor.DataType;

/**
 * BatchNormalization Operator v1
 * 
 * <p>
 * Carries out batch normalization as described in the paper
 * https://arxiv.org/abs/1502.03167. Depending on the mode it is being run,
 * there are multiple cases for the number of outputs, which we list below:
 * 
 * Output case #1: Y, mean, var, saved_mean, saved_var (training mode) Output
 * case #2: Y (test mode)
 * 
 * @author HarryLee {@literal <formaten@qq.com>}
 * @version 1
 * @since Version 1 of the default ONNX operator set
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Changelog.md#BatchNormalization-1">
 *      ONNX.Changelog.md</a>
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Operators.md#BatchNormalization">
 *      ONNX.Operators.md</a>
 */
public interface BatchNormalizationV1 extends AiOnnxOperatorV1 {

	public static final String OP_TYPE = "BatchNormalization";

	/**
	 * Constrain input and output types to float tensors.
	 */
	public static final TypeConstraint TPYE_CONSTRAINT_T = new TypeConstraint(DataType.floatTypes());

	@Override
	public default OperatorStatus getStatus() {
		return OperatorStatus.STABLE;
	}

	@Override
	public default String getOpType() {
		return OP_TYPE;
	}

	/**
	 * Inputs for operator execution (forward & backward)
	 *
	 * @param <T_TENSOR>
	 *            The backend tensor object.
	 */
	class BatchNormalizationInputsV1<T_TENSOR> extends OperatorInputs<T_TENSOR> {

		public static final String ATTR_EPSILON = "epsilon";

		public static final String ATTR_IS_TEST = "is_test";

		public static final String ATTR_MOMENTUM = "momentum";

		public static final String ATTR_SPATIAL = "spatial";

		public static final String ATTR_CONSUMED_INPUTS = "consumed_inputs";

		private AttributeField<List<Long>> consumedInputsField;

		private AttributeField<Float> epsilonField;

		private AttributeField<Long> isTestField;

		private AttributeField<Float> momentumField;

		private AttributeField<Long> spatialField;

		private InputField<T_TENSOR> xField;

		private InputField<T_TENSOR> scaleField;

		private InputField<T_TENSOR> bField;

		private InputField<T_TENSOR> meanField;

		private InputField<T_TENSOR> varField;

		/**
		 * Executes operator
		 * 
		 * @param x
		 *            The input 4-dimensional tensor of shape NCHW.
		 * @param scale
		 *            The scale as a 1-dimensional tensor of size C to be
		 *            applied to the output.
		 * @param b
		 *            The bias as a 1-dimensional tensor of size C to be applied
		 *            to the output.
		 * @param mean
		 *            The running mean (training) or the estimated mean
		 *            (testing) as a 1-dimensional tensor of size C.
		 * @param var
		 *            The running variance (training) or the estimated variance
		 *            (testing) as a 1-dimensional tensor of size C.
		 * @param consumedInputs
		 *            legacy optimization attribute.
		 * @param epsilon
		 *            The epsilon value to use to avoid division by zero,
		 *            default is 1e-5f.
		 * @param isTest
		 *            If set to nonzero, run spatial batch normalization in test
		 *            mode, default is 0.
		 * @param momentum
		 *            Factor used in computing the running mean and
		 *            variance.e.g., running_mean = running_mean * momentum +
		 *            mean * (1 - momentum), default is 0.9f.
		 * @param spatial
		 *            If true, compute the mean and variance across all spatial
		 *            elements If false, compute the mean and variance across
		 *            per feature.Default is 1.
		 * @return
		 * 
		 * 		<pre>
		 * Output case #1: Y, mean, var, saved_mean, saved_var (training mode)
		 *         </pre>
		 * 
		 *         <pre>
		 * Output case #2: Y (test mode)
		 *         </pre>
		 */
		public BatchNormalizationInputsV1(Node node, Inputs inputs) {
			super(node, inputs);

			//
			// Legacy optimization attribute.
			//
			consumedInputsField = new AttributeField<List<Long>>(super.attrs, ATTR_CONSUMED_INPUTS, IntsAttribute.class,
					null, false);

			//
			// The epsilon value to use to avoid division by zero, default is
			// 1e-5f.
			//
			epsilonField = new AttributeField<Float>(super.attrs, ATTR_EPSILON, FloatAttribute.class, 1e-5f, true);

			//
			// If set to nonzero, run spatial batch normalization in test mode,
			// default is 0.
			//
			isTestField = new AttributeField<Long>(super.attrs, ATTR_IS_TEST, IntAttribute.class, 0L, true);

			//
			// Factor used in computing the running mean and variance.e.g.,
			// running_mean = running_mean * momentum + mean * (1 - momentum),
			// default is 0.9f.
			//
			momentumField = new AttributeField<Float>(super.attrs, ATTR_MOMENTUM, FloatAttribute.class, 0.9f, true);

			//
			// If true, compute the mean and variance across all spatial
			// elements If
			// false, compute the mean and variance across per feature.Default
			// is 1.
			//
			spatialField = new AttributeField<Long>(super.attrs, ATTR_SPATIAL, IntAttribute.class, 1L, true);

			//
			// The input 4-dimensional tensor of shape NCHW.
			//
			xField = new InputField<T_TENSOR>(this, TPYE_CONSTRAINT_T, super.inputArray[0]);

			//
			// The scale as a 1-dimensional tensor of size C to be applied to
			// the output.
			//
			scaleField = new InputField<T_TENSOR>(this, TPYE_CONSTRAINT_T, super.inputArray[1]);

			//
			// The bias as a 1-dimensional tensor of size C to be applied to the
			// output.
			//
			bField = new InputField<T_TENSOR>(this, TPYE_CONSTRAINT_T, super.inputArray[2]);

			//
			// The running mean (training) or the estimated mean (testing) as a
			// 1-dimensional tensor of size C.
			//
			meanField = new InputField<T_TENSOR>(this, TPYE_CONSTRAINT_T, super.inputArray[3]);

			//
			// The running variance (training) or the estimated variance
			// (testing) as a 1-dimensional tensor of size C.
			//
			varField = new InputField<T_TENSOR>(this, TPYE_CONSTRAINT_T, super.inputArray[4]);
		}

		public T_TENSOR getX() {
			return xField.getData();
		}

		public List<Long> getConsumedInputs() {
			return consumedInputsField.getData();
		}

		public Float getEpsilon() {
			return epsilonField.getData();
		}

		public Boolean isTest() {
			return isTestField.getData() != 0L;
		}

		public Float getMomentum() {
			return momentumField.getData();
		}

		public Boolean isSpatial() {
			return spatialField.getData() == 1L;
		}

		public T_TENSOR getScale() {
			return scaleField.getData();
		}

		public T_TENSOR getB() {
			return bField.getData();
		}

		public T_TENSOR getMean() {
			return meanField.getData();
		}

		public T_TENSOR getVar() {
			return varField.getData();
		}

	}

	/**
	 * Outputs for operator execution (forward & backward)
	 *
	 * @param <T_TENSOR>
	 *            The backend tensor object.
	 */
	class BatchNormalizationOutputsV1<T_TENSOR> extends MultiOperatorOutputs<T_TENSOR> {

		protected Field<T_TENSOR> yField;

		protected Field<T_TENSOR> meanField;

		protected Field<T_TENSOR> varField;

		protected Field<T_TENSOR> savedMeanField;

		protected Field<T_TENSOR> savedVarField;

		public BatchNormalizationOutputsV1(OperatorOutputs<T_TENSOR> operatorOutputs) {
			//
			// Attention order of fields initialization
			//
			this.yField = new OutputField<T_TENSOR>(this, TPYE_CONSTRAINT_T, operatorOutputs.get(0), false);
			this.meanField = new OutputField<T_TENSOR>(this, TPYE_CONSTRAINT_T, operatorOutputs.get(1), true);
			this.varField = new OutputField<T_TENSOR>(this, TPYE_CONSTRAINT_T, operatorOutputs.get(2), true);
			this.savedMeanField = new OutputField<T_TENSOR>(this, TPYE_CONSTRAINT_T, operatorOutputs.get(3), true);
			this.savedVarField = new OutputField<T_TENSOR>(this, TPYE_CONSTRAINT_T, operatorOutputs.get(4), true);
		}

		public BatchNormalizationOutputsV1(T_TENSOR y) {
			this(y, null, null, null, null);
		}

		public BatchNormalizationOutputsV1(T_TENSOR data, T_TENSOR mean, T_TENSOR var, T_TENSOR savedMean,
				T_TENSOR savedVar) {
			//
			// Attention order of fields initialization
			//
			this.yField = new OutputField<T_TENSOR>(this, TPYE_CONSTRAINT_T, data, false);
			this.meanField = new OutputField<T_TENSOR>(this, TPYE_CONSTRAINT_T, mean, true);
			this.varField = new OutputField<T_TENSOR>(this, TPYE_CONSTRAINT_T, var, true);
			this.savedMeanField = new OutputField<T_TENSOR>(this, TPYE_CONSTRAINT_T, savedMean, true);
			this.savedVarField = new OutputField<T_TENSOR>(this, TPYE_CONSTRAINT_T, savedVar, true);
		}

	}

}