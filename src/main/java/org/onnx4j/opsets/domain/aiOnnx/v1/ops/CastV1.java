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

import org.onnx4j.Inputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.model.graph.node.attributes.StringAttribute;
import org.onnx4j.opsets.domain.aiOnnx.v1.AiOnnxOperatorV1;
import org.onnx4j.opsets.operator.Field;
import org.onnx4j.opsets.operator.Field.TypeConstraint;
import org.onnx4j.opsets.operator.OperatorInputs;
import org.onnx4j.opsets.operator.fields.AttributeField;
import org.onnx4j.opsets.operator.fields.InputField;
import org.onnx4j.opsets.operator.output.SingleOperatorOutputs;
import org.onnx4j.tensor.DataType;

/**
 * Cast Operator v1
 * 
 * <p>
 * The operator casts the elements of a given input tensor to a data type
 * specified by the 'to' argument and returns an output tensor of the same size
 * in the converted type. The 'to' argument must be one of the data types
 * specified in the 'DataType' enum field in the TensorProto message. NOTE:
 * Casting to and from strings is not supported yet.
 * 
 * @author HarryLee {@literal <formaten@qq.com>}
 * @version 1
 * @since Version 1 of the default ONNX operator set
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Cast-1">ONNX
 *      .Changelog.md</a>
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Operators.md#Cast">ONNX.
 *      Operators.md</a>
 */
public interface CastV1 extends AiOnnxOperatorV1 {

	public static final String OP_TYPE = "Cast";

	/**
	 * Constrain input types. Casting from strings and complex are not
	 * supported.
	 */
	public static final TypeConstraint TPYE_CONSTRAINT_T1 = new TypeConstraint(
			DataType.allTypesWithoutStringAndComplex());

	/**
	 * Constrain input types. Casting from strings and complex are not
	 * supported.
	 */
	public static final TypeConstraint TPYE_CONSTRAINT_T2 = new TypeConstraint(
			DataType.allTypesWithoutStringAndComplex());

	/**
	 * Executes operator
	 * 
	 * @param input
	 *            Input tensor to be cast.
	 * @param to
	 *            The data type to which the elements of the input tensor are
	 *            cast. Strictly must be one of the types from DataType enum in
	 *            TensorProto
	 * @return Output tensor with the same shape as input with type specified by
	 *         the 'to' argument
	 */
	//public T_TENSOR cast(T_TENSOR input, String to);

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
	class CastInputsV1<T_TENSOR> extends OperatorInputs<T_TENSOR> {

		public static final String ATTR_TO = "to";

		protected Field<T_TENSOR> inputField;

		protected Field<String> toField;

		public CastInputsV1(Node node, Inputs inputs) {
			super(node, inputs);

			inputField = new InputField<T_TENSOR>(this, TPYE_CONSTRAINT_T1, super.inputArray[0]);
			toField = new AttributeField<String>(super.attrs, ATTR_TO, StringAttribute.class, null, true);
		}

		public T_TENSOR getInput() {
			return inputField.getData();
		}

		public String getTo() {
			return toField.getData();
		}

	}

	/**
	 * Outputs for operator execution (forward & backward)
	 *
	 * @param <T_TENSOR>
	 *            The backend tensor object.
	 */
	class CastOutputV1<T_TENSOR> extends SingleOperatorOutputs<T_TENSOR> {

		public CastOutputV1(T_TENSOR output) {
			super(output);
		}

		@Override
		public TypeConstraint getTypeConstraint() {
			return TPYE_CONSTRAINT_T2;
		}

	}

}