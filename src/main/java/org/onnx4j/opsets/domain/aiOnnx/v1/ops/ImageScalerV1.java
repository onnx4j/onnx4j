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
import org.onnx4j.model.graph.node.attributes.FloatsAttribute;
import org.onnx4j.opsets.domain.aiOnnx.v1.AiOnnxOperatorV1;
import org.onnx4j.opsets.operator.Field;
import org.onnx4j.opsets.operator.Field.TypeConstraint;
import org.onnx4j.opsets.operator.OperatorInputs;
import org.onnx4j.opsets.operator.fields.AttributeField;
import org.onnx4j.opsets.operator.fields.InputField;
import org.onnx4j.opsets.operator.output.SingleOperatorOutputs;
import org.onnx4j.tensor.DataType;

/**
 * {@value #OP_TYPE} Operator
 * 
 * @deprecated
 * @author HarryLee {@literal <formaten@qq.com>}
 * @version 1
 * @since Version 1 of the default ONNX operator set
 */
@Deprecated
public interface ImageScalerV1 extends AiOnnxOperatorV1 {

	public static final String OP_TYPE = "ImageScaler";

	/**
	 * Constrain input and output types to float tensors.
	 */
	public static final TypeConstraint TPYE_CONSTRAINT_T = new TypeConstraint(DataType.floatTypes());

	//public abstract T_TENSOR scale(T_TENSOR input, Float scale, List<Float> bias);

	@Override
	public default OperatorStatus getStatus() {
		return OperatorStatus.REMOVED;
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
	class ImageScalerInputsV1<T_TENSOR> extends OperatorInputs<T_TENSOR> {

		public static final String ATTR_SCALE = "scale";

		public static final String ATTR_BIAS = "bias";

		protected Field<T_TENSOR> inputField;

		protected Field<List<Float>> biasField;

		protected Field<Float> scaleField;

		public ImageScalerInputsV1(Node node, Inputs inputs) {
			super(node, inputs);

			inputField = new InputField<T_TENSOR>(this, TPYE_CONSTRAINT_T, super.inputArray[0]);

			biasField = new AttributeField<List<Float>>(super.attrs, ATTR_BIAS, FloatsAttribute.class, null, false);

			scaleField = new AttributeField<Float>(super.attrs, ATTR_SCALE, FloatAttribute.class, 0F, true);
		}

		public T_TENSOR getInput() {
			return inputField.getData();
		}

		public List<Float> getBias() {
			return biasField.getData();
		}

		public Float getScale() {
			return scaleField.getData();
		}

	}

	/**
	 * Outputs for operator execution (forward & backward) s
	 * 
	 * @param <T_TENSOR>
	 *            The backend tensor object.
	 */
	class ImageScalerOutputV1<T_TENSOR> extends SingleOperatorOutputs<T_TENSOR> {

		public ImageScalerOutputV1(T_TENSOR output) {
			super(output);
		}

		@Override
		public TypeConstraint getTypeConstraint() {
			return TPYE_CONSTRAINT_T;
		}

	}

}