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
import org.onnx4j.model.graph.node.attributes.IntsAttribute;
import org.onnx4j.opsets.domain.aiOnnx.v1.AiOnnxOperatorV1;
import org.onnx4j.opsets.operator.Field;
import org.onnx4j.opsets.operator.Field.TypeConstraint;
import org.onnx4j.opsets.operator.OperatorInputs;
import org.onnx4j.opsets.operator.fields.AttributeField;
import org.onnx4j.opsets.operator.fields.InputField;
import org.onnx4j.opsets.operator.output.SingleOperatorOutputs;
import org.onnx4j.tensor.DataType;

/**
 * Reshape Operator v1
 * 
 * <p>
 * Operator Reshape the input tensor similar to numpy.reshape. It takes a tensor
 * as input and an argument shape. It outputs the reshaped tensor. At most one
 * dimension of the new shape can be -1. In this case, the value is inferred
 * from the size of the tensor and the remaining dimensions. A dimension could
 * also be 0, in which case the actual dimension value is unchanged (i.e. taken
 * from the input tensor).
 * 
 * @author HarryLee {@literal <formaten@qq.com>}
 * @version 1
 * @since Version 1 of the default ONNX operator set
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Reshape-1">
 *      ONNX.Changelog.md</a>
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Operators.md#Reshape">
 *      ONNX.Operators.md</a>
 */
public interface ReshapeV1 extends AiOnnxOperatorV1 {

	public static final String OP_TYPE = "Reshape";

	/**
	 * Constrain input and output types to float tensors.
	 */
	public static final TypeConstraint TPYE_CONSTRAINT_T = new TypeConstraint(DataType.floatTypes());

	/**
	 * Executes operator
	 * 
	 * @param data
	 *            An input tensor
	 * @param shape
	 *            New shape
	 * @param consumedInputs
	 *            legacy optimization attribute
	 * @return Reshaped data
	 */
	// public abstract T_TENSOR reshape(T_TENSOR data, List<Long> shape,
	// List<Long> consumedInputs);

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
	class ReshapeInputsV1<T_TENSOR> extends OperatorInputs<T_TENSOR> {

		//
		// New shape
		//
		public static final String ATTR_SHAPE = "shape";

		//
		// Legacy optimization attribute.
		//
		public static final String ATTR_CONSUMED_INPUTS = "consumed_inputs";

		private Field<T_TENSOR> dataField;

		private Field<List<Long>> shapeField;

		private Field<List<Long>> consumedInputsField;

		public ReshapeInputsV1(Node node, Inputs inputs) {
			super(node, inputs);

			this.dataField = new InputField<T_TENSOR>(this, TPYE_CONSTRAINT_T, super.inputArray[0]);
			this.shapeField = new AttributeField<List<Long>>(super.attrs, ATTR_SHAPE, IntsAttribute.class, null, false);
			this.consumedInputsField = new AttributeField<List<Long>>(super.attrs, ATTR_CONSUMED_INPUTS,
					IntsAttribute.class, null, false);
		}

		public T_TENSOR getData() {
			return dataField.getData();
		}

		public List<Long> getShape() {
			return shapeField.getData();
		}

		public List<Long> getConsumedInputs() {
			return consumedInputsField.getData();
		}

	}

	/**
	 * Outputs for operator execution (forward & backward)
	 * 
	 * @param <T_TENSOR>
	 *            The backend tensor object.
	 */
	class ReshapeOutputV1<T_TENSOR> extends SingleOperatorOutputs<T_TENSOR> {

		public ReshapeOutputV1(T_TENSOR output) {
			super(output);
		}

		@Override
		public TypeConstraint getTypeConstraint() {
			return TPYE_CONSTRAINT_T;
		}

	}

}