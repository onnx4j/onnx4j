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
import org.onnx4j.model.graph.node.attributes.IntAttribute;
import org.onnx4j.model.graph.node.attributes.IntsAttribute;
import org.onnx4j.model.graph.node.attributes.StringAttribute;
import org.onnx4j.opsets.operator.Field.TypeConstraint;
import org.onnx4j.opsets.domain.aiOnnx.v1.AiOnnxOperatorV1;
import org.onnx4j.opsets.operator.OperatorInputs;
import org.onnx4j.opsets.operator.fields.AttributeField;
import org.onnx4j.opsets.operator.fields.InputField;
import org.onnx4j.opsets.operator.output.SingleOperatorOutputs;
import org.onnx4j.tensor.DataType;

/**
 * Conv Operator v1
 * 
 * <p>
 * The convolution operator consumes an input tensor and a filter, and computes
 * the output.
 * 
 * @author HarryLee {@literal <formaten@qq.com>}
 * @version 1
 * @since Version 1 of the default ONNX operator set
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Conv-1">ONNX
 *      .Changelog.md</a>
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Operators.md#Conv">ONNX.
 *      Operators.md</a>
 */
public interface ConvV1 extends AiOnnxOperatorV1 {

	public static final String OP_TYPE = "Conv";

	/**
	 * Constrain input and output types to float tensors.
	 */
	public static final TypeConstraint TPYE_CONSTRAINT_T = new TypeConstraint(DataType.floatTypes());

	/**
	 * Executes operator
	 * 
	 * @param x
	 *            Input data tensor from previous layer; has size (N x C x H x
	 *            W), where N is the batch size, C is the number of channels,
	 *            and H and W are the height and width. Note that this is for
	 *            the 2D image. Otherwise the size is (N x C x D1 x D2 ... x
	 *            Dn). Optionally, if dimension denotation is in effect, the
	 *            operation expects input data tensor to arrive with the
	 *            dimension denotation of [DATA_BATCH, DATA_CHANNEL,
	 *            DATA_FEATURE, DATA_FEATURE ...].
	 * @param w
	 *            The weight tensor that will be used in the convolutions; has
	 *            size (M x C/group x kH x kW), where C is the number of
	 *            channels, and kH and kW are the height and width of the
	 *            kernel, and M is the number of feature maps. For more than 2
	 *            dimensions, the kernel shape will be (M x C/group x k1 x k2 x
	 *            ... x kn), where (k1 x k2 x ... kn) is the dimension of the
	 *            kernel. Optionally, if dimension denotation is in effect, the
	 *            operation expects the weight tensor to arrive with the
	 *            dimension denotation of [FILTER_OUT_CHANNEL,
	 *            FILTER_IN_CHANNEL, FILTER_SPATIAL, FILTER_SPATIAL ...].
	 *            X.shape[1] == (W.shape[1] * group) == C (assuming zero based
	 *            indices for the shape array). Or in other words
	 *            FILTER_IN_CHANNEL should be equal to DATA_CHANNEL.
	 * @param b
	 *            Optional 1D bias to be added to the convolution, has size of
	 *            M.
	 * @param autoPad
	 * @param dilations
	 * @param group
	 * @param kernelShape
	 * @param pads
	 * @param strides
	 * @return
	 */
	/*
	 * public abstract T_TENSOR conv(T_TENSOR x, T_TENSOR w, T_TENSOR b, String
	 * autoPad, List<Long> dilations, Long group, List<Long> kernelShape,
	 * List<Long> pads, List<Long> strides);
	 */

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
	class ConvInputsV1<T_TENSOR> extends OperatorInputs<T_TENSOR> {

		public static final String ATTR_AUTO_PAD = "auto_pad";

		public static final String ATTR_KERNEL_SHAPE = "kernel_shape";

		public static final String ATTR_PADS = "pads";

		public static final String ATTR_STRIDES = "strides";

		public static final String ATTR_GROUP = "group";

		public static final String ATTR_DILATIONS = "dilations";

		private AttributeField<String> autoPadField;

		private AttributeField<List<Long>> kernelShapeField;

		private AttributeField<List<Long>> dilationsField;

		private AttributeField<Long> groupField;

		private AttributeField<List<Long>> padsField;

		private AttributeField<List<Long>> stridesField;

		private InputField<T_TENSOR> xField;

		private InputField<T_TENSOR> bField;

		private InputField<T_TENSOR> wField;

		public ConvInputsV1(Node node, Inputs inputs) {
			super(node, inputs);

			//
			// string (default is NOTSET)
			//
			autoPadField = new AttributeField<String>(super.attrs, ATTR_AUTO_PAD, StringAttribute.class, "NOTSET",
					false);

			//
			// list of ints
			//
			dilationsField = new AttributeField<List<Long>>(super.attrs, ATTR_DILATIONS, IntsAttribute.class, null,
					true);

			//
			// int (default is 1)
			//
			groupField = new AttributeField<Long>(super.attrs, ATTR_GROUP, IntAttribute.class, 1L, true);

			//
			// list of ints (required)
			//
			kernelShapeField = new AttributeField<List<Long>>(super.attrs, ATTR_KERNEL_SHAPE, IntsAttribute.class, null,
					false);

			//
			// list of ints
			//
			padsField = new AttributeField<List<Long>>(super.attrs, ATTR_PADS, IntsAttribute.class, null, false);

			//
			// list of ints
			//
			stridesField = new AttributeField<List<Long>>(super.attrs, ATTR_STRIDES, IntsAttribute.class, null, false);

			//
			// The input 4-dimensional tensor of shape NCHW.
			//
			xField = new InputField<T_TENSOR>(this, TPYE_CONSTRAINT_T, super.inputArray[0]);

			//
			// The scale as a 1-dimensional tensor of size C to be applied to
			// the output.
			//
			wField = new InputField<T_TENSOR>(this, TPYE_CONSTRAINT_T, super.inputArray[1]);

			//
			// The bias as a 1-dimensional tensor of size C to be applied to the
			// output.
			//
			bField = super.inputArray.length > 2
					? new InputField<T_TENSOR>(this, TPYE_CONSTRAINT_T, super.inputArray[2]) : null;
		}

		public T_TENSOR getX() {
			return xField.getData();
		}

		public String getAutoPad() {
			return autoPadField.getData();
		}

		public List<Long> getKernelShape() {
			return kernelShapeField.getData();
		}

		public List<Long> getDilations() {
			return dilationsField.getData();
		}

		public Long getGroup() {
			return groupField.getData();
		}

		public List<Long> getPads() {
			return padsField.getData();
		}

		public List<Long> getStrides() {
			return stridesField.getData();
		}

		public T_TENSOR getW() {
			return wField.getData();
		}

		public T_TENSOR getB() {
			return (bField == null) ? null : bField.getData();
		}

	}

	/**
	 * Outputs for operator execution (forward & backward)
	 *
	 * @param <T_TENSOR>
	 *            The backend tensor object.
	 */
	class ConvOutputV1<T_TENSOR> extends SingleOperatorOutputs<T_TENSOR> {

		public ConvOutputV1(T_TENSOR output) {
			super(output);
		}

		@Override
		public TypeConstraint getTypeConstraint() {
			return TPYE_CONSTRAINT_T;
		}

	}

}